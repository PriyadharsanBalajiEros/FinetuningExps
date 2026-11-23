"""
End-to-End Inference, True Renderer-in-the-Loop (RITL) Self-Repair & MP4 Video Generator.
=========================================================================================
Loads our two-stage 16-bit fine-tuned model from HuggingFace Hub:
  1. Base Backbone: `Qwen/Qwen2.5-Coder-7B-Instruct` (Native FP16)
  2. Stage-1 Gold SFT Adapter: `PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit` (`subfolder="best"`)
  3. Stage-2 DPO Aligned Adapter: `PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit`
Generates `<think>` spatial layout planning + runnable ManimCE v0.19+ Python code,
renders the actual `.mp4` animation via `manim`, and if either static AST verification
OR actual Manim video rendering fails on Pass 1, feeds the exact traceback back to
the model to self-repair and render the final `.mp4` video!
"""

import argparse
import glob
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
    sft_repo_id: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-V3-SFT-16bit",
    dpo_repo_id: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-V3-DPO-16bit",
    hf_token: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load 16-bit base model and merge Stage-1 Gold SFT (`best`) + Stage-2 DPO adapters."""
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
    print("[Ready] 16-Bit ManimCE SFT+DPO Model loaded and merged!")
    return model, tokenizer


def render_manim_video(
    py_path: str,
    scene_name: str,
    quality: str = "m",
    media_dir: str = "media",
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Render `.mp4` video using `manim -q<quality> --media_dir <media_dir> <py_path> <scene_name>`.
    Includes both stdout LaTeX error logs and stderr tracebacks if rendering fails.
    """
    q_flag = f"-q{quality}" if not quality.startswith("-") else quality
    cmd = ["manim", "render", q_flag, "--media_dir", media_dir, py_path, scene_name]
    print(f"\n[Manim Renderer] Executing: {' '.join(cmd)} ...")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if proc.returncode == 0:
            mp4_candidates = glob.glob(os.path.join(media_dir, "videos", "**", f"{scene_name}.mp4"), recursive=True)
            if not mp4_candidates:
                mp4_candidates = glob.glob(os.path.join(media_dir, "**", "*.mp4"), recursive=True)
            mp4_path = max(mp4_candidates, key=os.path.getmtime) if mp4_candidates else None
            return True, mp4_path, None
        combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        tail_err = "\n".join(combined.splitlines()[-45:])
        return False, None, tail_err
    except FileNotFoundError:
        return False, None, "Manim CLI not found. Install via `pip install manim`."
    except subprocess.TimeoutExpired:
        return False, None, "TimeoutError: Rendering exceeded 240s."


@torch.no_grad()
def generate_and_render_manim(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 2560,
    output_py_path: str = "generated_scene.py",
    render_quality: str = "m",
    media_dir: str = "media",
) -> Optional[str]:
    """
    Generate ManimCE code, render the `.mp4` video, and automatically trigger
    1-turn Renderer-in-the-Loop (RITL) self-repair if either static AST validation
    OR actual Manim video rendering throws an error.
    """
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

    print(f"\n[Pass 1 Generation] Prompt: {prompt!r} ...")
    raw_reply_1 = _sample(messages)
    vres = verify_and_canonicalize(raw_reply_1)

    render_error = None
    if vres.is_valid:
        with open(output_py_path, "w", encoding="utf-8") as f:
            f.write(vres.canonical_code + "\n")
        ok, mp4_path, render_error = render_manim_video(
            output_py_path, vres.scene_name or "MainScene", quality=render_quality, media_dir=media_dir
        )
        if ok and mp4_path:
            print(f"\n[SUCCESS] Video rendered on Pass 1: {mp4_path}")
            return mp4_path
    else:
        render_error = f"Static AST/API Error: {vres.rejection_reason}"

    # --- Pass 2: Renderer-in-the-Loop (RITL) Self-Repair ---
    print(f"\n[RITL Self-Repair] Feeding compiler traceback back to model for 1-turn repair:\n{render_error}")
    repair_msg = (
        f"Running the generated Manim script produced the following compiler/runtime error:\n"
        f"```text\n{render_error}\n```\n\n"
        "Please fix the bug and provide the complete corrected ManimCE v0.19+ Python script."
    )
    messages.append({"role": "assistant", "content": raw_reply_1})
    messages.append({"role": "user", "content": repair_msg})

    raw_reply_2 = _sample(messages)
    vres_2 = verify_and_canonicalize(raw_reply_2)
    if vres_2.is_valid:
        with open(output_py_path, "w", encoding="utf-8") as f:
            f.write(vres_2.canonical_code + "\n")
        ok_2, mp4_path_2, err_2 = render_manim_video(
            output_py_path, vres_2.scene_name or "MainScene", quality=render_quality, media_dir=media_dir
        )
        if ok_2 and mp4_path_2:
            print(f"\n[SUCCESS] Video rendered after Pass 2 RITL Self-Repair: {mp4_path_2}")
            return mp4_path_2
        print(f"\n[WARN] Pass 2 render error: {err_2}")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and render ManimCE MP4 videos with our 16-bit SFT+DPO model")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Create an elegant 3Blue1Brown-style ManimCE animation visually proving the Pythagorean theorem a^2 + b^2 = c^2 with a right triangle, colored squares on each side, and step-by-step equations.",
    )
    parser.add_argument("--output", type=str, default="generated_scene.py")
    parser.add_argument("--render", type=str, default="m", choices=["l", "m", "h", "k"])
    parser.add_argument("--max_new_tokens", type=int, default=2560)
    args = parser.parse_args()

    model, tokenizer = load_manimce_16bit_model(hf_token=os.environ.get("HF_TOKEN"))
    generate_and_render_manim(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        output_py_path=args.output,
        render_quality=args.render,
    )
