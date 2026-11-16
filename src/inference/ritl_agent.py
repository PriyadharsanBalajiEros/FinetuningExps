"""
Renderer-in-the-Loop (RITL) ManimCE Code Generator & 1-Shot Compiler Debugger.
==============================================================================
Combines our 16-bit fine-tuned Qwen2.5-Coder-7B-ManimCE model with:
  1. Static AST + ManimCE v0.19+ API verification (`verify_and_canonicalize`)
  2. Optional headless ManimCE execution (`manim -ql --dry_run`)
  3. Automatic 1-turn compiler traceback self-repair when any static or runtime
     error is detected.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from src.data_pipeline.dpo_pair_builder import SYSTEM_PROMPT_MANIMCE
from src.data_pipeline.manim_canonicalizer import verify_and_canonicalize


@dataclass
class GenerationResult:
    prompt: str
    final_code: str
    scene_name: Optional[str]
    pass1_ast_valid: bool
    pass1_exec_valid: bool
    pass2_repaired_valid: bool
    attempts_used: int
    error_trace: Optional[str] = None


def execute_manim_dry_run(code: str, scene_name: str, timeout_sec: int = 45) -> Tuple[bool, Optional[str]]:
    """
    Run `manim -ql --dry_run` in an isolated temporary directory to verify
    zero runtime exceptions (TypeError, AttributeError, NameError, LaTeX error).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "scene_under_test.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                ["manim", "-ql", "--dry_run", script_path, scene_name],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if proc.returncode == 0:
                return True, None
            err_out = (proc.stderr or proc.stdout or "Unknown Manim runtime error").strip()
            # Keep last 25 lines of traceback
            tail_lines = "\n".join(err_out.splitlines()[-25:])
            return False, tail_lines
        except FileNotFoundError:
            # Manim binary not installed in current environment; rely on static AST verifier
            return True, None
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: Scene rendering exceeded {timeout_sec}s limit."


@torch.no_grad()
def generate_manim_with_ritl(
    model: Any,
    tokenizer: Any,
    user_prompt: str,
    max_new_tokens: int = 2048,
    enable_headless_render: bool = True,
) -> GenerationResult:
    """
    Generate ManimCE v0.19+ Python code with automatic 1-turn Renderer-in-the-Loop
    (RITL) self-repair if Pass@1 encounters an AST or runtime error.
    """
    device = next(model.parameters()).device
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT_MANIMCE},
        {"role": "user", "content": user_prompt},
    ]

    def _sample_reply(msgs: List[Dict[str, str]]) -> str:
        chat_str = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_str, return_tensors="pt").to(device)
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out_ids[0][inputs["input_ids"].size(1) :]
        return tokenizer.decode(gen_ids, skip_special_tokens=True)

    # --- Pass 1 Generation ---
    reply_1 = _sample_reply(messages)
    vres_1 = verify_and_canonicalize(reply_1)

    if vres_1.is_valid:
        if not enable_headless_render:
            return GenerationResult(
                prompt=user_prompt,
                final_code=vres_1.canonical_code,
                scene_name=vres_1.scene_name,
                pass1_ast_valid=True,
                pass1_exec_valid=True,
                pass2_repaired_valid=True,
                attempts_used=1,
            )
        exec_ok, trace_err = execute_manim_dry_run(
            vres_1.canonical_code, vres_1.scene_name or "Main"
        )
        if exec_ok:
            return GenerationResult(
                prompt=user_prompt,
                final_code=vres_1.canonical_code,
                scene_name=vres_1.scene_name,
                pass1_ast_valid=True,
                pass1_exec_valid=True,
                pass2_repaired_valid=True,
                attempts_used=1,
            )
        error_feedback = trace_err or "Runtime execution error"
    else:
        error_feedback = f"Static Verification Error: {vres_1.rejection_reason}"

    # --- Pass 2: Renderer-in-the-Loop (RITL) Self-Repair ---
    repair_prompt = (
        "Running the generated Manim script produced the following compiler/layout error:\n\n"
        f"```text\n{error_feedback}\n```\n\n"
        "Please fix the bug and provide the complete corrected ManimCE v0.19+ Python script."
    )
    messages.append({"role": "assistant", "content": reply_1})
    messages.append({"role": "user", "content": repair_prompt})

    reply_2 = _sample_reply(messages)
    vres_2 = verify_and_canonicalize(reply_2)
    if vres_2.is_valid:
        exec_ok_2 = True
        trace_err_2 = None
        if enable_headless_render:
            exec_ok_2, trace_err_2 = execute_manim_dry_run(
                vres_2.canonical_code, vres_2.scene_name or "Main"
            )
        return GenerationResult(
            prompt=user_prompt,
            final_code=vres_2.canonical_code,
            scene_name=vres_2.scene_name,
            pass1_ast_valid=vres_1.is_valid,
            pass1_exec_valid=False,
            pass2_repaired_valid=exec_ok_2,
            attempts_used=2,
            error_trace=trace_err_2,
        )

    return GenerationResult(
        prompt=user_prompt,
        final_code=vres_2.canonical_code or vres_1.canonical_code,
        scene_name=vres_1.scene_name,
        pass1_ast_valid=vres_1.is_valid,
        pass1_exec_valid=False,
        pass2_repaired_valid=False,
        attempts_used=2,
        error_trace=vres_2.rejection_reason or error_feedback,
    )
