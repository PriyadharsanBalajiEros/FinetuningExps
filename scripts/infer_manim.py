"""
Standalone Inference & Self-Repair CLI for Qwen2.5-Coder-7B-ManimCE (16-Bit SFT + DPO).
=======================================================================================
Loads our two-stage 16-bit fine-tuned model from HuggingFace Hub:
  1. Base Backbone: `Qwen/Qwen2.5-Coder-7B-Instruct` (Native FP16)
  2. Stage-1 Gold SFT Adapter: `PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit` (`subfolder="best"`)
  3. Stage-2 DPO Aligned Adapter: `PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit`
Generates `<think>` spatial layout planning + runnable ManimCE v0.19+ Python code,
verifies the AST & API compliance via `verify_and_canonicalize`, runs 1-turn
Renderer-in-the-Loop (RITL) self-repair if needed, saves the `.py` scene file,
and optionally renders the `.mp4` animation via `manim -pqh` / `manim -pql`.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import peft.import_utils
import peft.tuners.lora.torchao
peft.import_utils.is_torchao_available = lambda: False
peft.tuners.lora.torchao.is_torchao_available = lambda: False
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.dpo_pair_builder import SYSTEM_PROMPT_MANIMCE
from src.data_pipeline.manim_canonicalizer import verify_and_canonicalize


def load_manimce_16bit_model(
    base_model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    sft_repo_id: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit",
    dpo_repo_id: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit",
    hf_token: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the 16-bit base model, merge the Gold Step-50 Stage-1 SFT adapter (`best/`),
    and merge the Stage-2 DPO adapter for fast KV-cached 16-bit inference.
    """
    print(f"[1/3] Loading tokenizer & 16-bit base model: {base_model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )

    print(f"[2/3] Merging Gold Stage-1 16-Bit SFT Adapter ({sft_repo_id}, subfolder='best') ...")
    model = PeftModel.from_pretrained(model, sft_repo_id, subfolder="best", token=hf_token)
    model = model.merge_and_unload()

    print(f"[3/3] Merging Stage-2 16-Bit DPO Aligned Adapter ({dpo_repo_id}) ...")
    model = PeftModel.from_pretrained(model, dpo_repo_id, token=hf_token)
    model = model.merge_and_unload()

    model.config.use_cache = True
    model.eval()
    print("[Ready] 16-Bit ManimCE SFT+DPO Model loaded and merged for fast KV-cached inference!")
    return model, tokenizer


@torch.no_grad()
def generate_manim_scene(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 2560,
    output_py_path: str = "generated_scene.py",
    render_quality: Optional[str] = None,
) -> str:
    """Generate, AST-verify, self-repair (if needed), save, and optionally render a ManimCE script."""
    device = next(model.parameters()).device
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_MANIMCE},
        {"role": "user", "content": prompt},
    ]

    def _sample(msgs):
        chat_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt").to(device)
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out_ids[0][inputs["input_ids"].size(1) :], skip_special_tokens=True)

    print(f"\n[Generating Pass@1] Prompt: {prompt!r} ...")
    raw_reply_1 = _sample(messages)
    vres = verify_and_canonicalize(raw_reply_1)

    if not vres.is_valid:
        print(f"[RITL Self-Repair] Pass@1 issue detected ({vres.rejection_reason}). Running 1-turn self-repair...")
        repair_msg = (
            f"Running the generated Manim script produced the following error:\n```text\n{vres.rejection_reason}\n```\n"
            "Please fix the error and output the complete runnable ManimCE v0.19+ Python script."
        )
        messages.append({"role": "assistant", "content": raw_reply_1})
        messages.append({"role": "user", "content": repair_msg})
        raw_reply_2 = _sample(messages)
        vres_2 = verify_and_canonicalize(raw_reply_2)
        if vres_2.is_valid:
            vres = vres_2

    final_code = vres.canonical_code
    scene_name = vres.scene_name or "MainScene"

    with open(output_py_path, "w", encoding="utf-8") as f:
        f.write(final_code + "\n")

    print(f"\n[Saved] Clean ManimCE v0.19+ script written to: {output_py_path} (Scene: {scene_name})")
    print("-" * 72)
    print(final_code)
    print("-" * 72)

    if render_quality:
        q_flag = f"-q{render_quality}" if not render_quality.startswith("-") else render_quality
        cmd = ["manim", q_flag, output_py_path, scene_name]
        print(f"\n[Rendering Video] Running: {' '.join(cmd)} ...")
        subprocess.run(cmd, check=False)

    return final_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ManimCE animations with our 16-bit SFT+DPO model")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Create an elegant 3Blue1Brown-style ManimCE animation visually proving the Pythagorean theorem a^2 + b^2 = c^2 with a right triangle, colored squares on each side, and step-by-step equations.",
        help="Natural language description of the mathematical animation to generate.",
    )
    parser.add_argument("--output", type=str, default="generated_scene.py", help="Output Python file path.")
    parser.add_argument(
        "--render",
        type=str,
        default=None,
        choices=["l", "m", "h", "k"],
        help="Optional Manim render quality ('l'=480p, 'm'=720p, 'h'=1080p, 'k'=4K) if Manim is installed.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2560)
    args = parser.parse_args()

    model, tokenizer = load_manimce_16bit_model(hf_token=os.environ.get("HF_TOKEN"))
    generate_manim_scene(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        output_py_path=args.output,
        render_quality=args.render,
    )

