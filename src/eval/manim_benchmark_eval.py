"""
ManimBench-v1 & Autoresearch-Manim Held-Out Benchmark Evaluator.
================================================================
Evaluates our 16-bit fine-tuned model on the held-out test split (`data/test.jsonl`)
from `PRB1/ManimCE-Master-SFT`, measuring:
  - Pass@1 Static AST & ManimCE v0.19+ API Compliance (%)
  - Pass@1 Headless ManimCE Execution Success (%)
  - Pass@2 Renderer-in-the-Loop (RITL) Self-Repair Execution Success (%)
  - Zero Deprecated Hallucination Rate (%)
"""

import argparse
import json
import os
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download

from src.inference.ritl_agent import generate_manim_with_ritl
from src.training.sft_trainer import (
    get_balanced_device_map,
    load_jsonl_from_hub_or_local,
)


def evaluate_manim_benchmark(
    base_model_id: str,
    adapter_repo_id: str,
    dataset_repo_id: str = "PRB1/ManimCE-Master-SFT",
    max_cases: int = 60,
    enable_headless_render: bool = True,
    hf_token: str = "",
    output_json: str = "outputs/manimbench_eval_results.json",
) -> Dict[str, Any]:
    """Run benchmark evaluation across held-out test prompts."""
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    device_map, max_memory = get_balanced_device_map()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
    )
    if adapter_repo_id:
        adapter_dir = snapshot_download(repo_id=adapter_repo_id, repo_type="model", token=hf_token)
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        model = base_model
    model.eval()

    test_rows = load_jsonl_from_hub_or_local(
        dataset_repo_id, "data/test.jsonl", "data/processed/eval_benchmark.jsonl", hf_token
    )[:max_cases]

    results: List[Dict[str, Any]] = []
    p1_ast_ok = 0
    p1_exec_ok = 0
    p2_exec_ok = 0

    print(f"Evaluating {len(test_rows)} held-out ManimBench / Autoresearch test cases ...")
    for idx, row in enumerate(test_rows):
        res = generate_manim_with_ritl(
            model,
            tokenizer,
            user_prompt=row["prompt"],
            enable_headless_render=enable_headless_render,
        )
        p1_ast_ok += int(res.pass1_ast_valid)
        p1_exec_ok += int(res.pass1_exec_valid)
        p2_exec_ok += int(res.pass2_repaired_valid)
        results.append(
            {
                "id": row.get("id", f"test_{idx}"),
                "source": row.get("source", ""),
                "pass1_ast_valid": res.pass1_ast_valid,
                "pass1_exec_valid": res.pass1_exec_valid,
                "pass2_repaired_valid": res.pass2_repaired_valid,
                "attempts_used": res.attempts_used,
                "error_trace": res.error_trace,
            }
        )
        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_rows):
            n = idx + 1
            print(
                f"  [{n:3d}/{len(test_rows):3d}] Pass@1 AST: {100*p1_ast_ok/n:.1f}% | "
                f"Pass@1 Exec: {100*p1_exec_ok/n:.1f}% | Pass@2 RITL: {100*p2_exec_ok/n:.1f}%"
            )

    total = max(1, len(test_rows))
    summary = {
        "base_model": base_model_id,
        "adapter_repo": adapter_repo_id,
        "num_cases": len(test_rows),
        "pass1_ast_valid_pct": round(100.0 * p1_ast_ok / total, 2),
        "pass1_exec_valid_pct": round(100.0 * p1_exec_ok / total, 2),
        "pass2_ritl_repaired_pct": round(100.0 * p2_exec_ok / total, 2),
        "details": results,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== BENCHMARK EVALUATION SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter_repo", type=str, default="PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit")
    parser.add_argument("--max_cases", type=int, default=60)
    args = parser.parse_args()
    evaluate_manim_benchmark(
        base_model_id=args.base_model,
        adapter_repo_id=args.adapter_repo,
        max_cases=args.max_cases,
        hf_token=os.environ.get("HF_TOKEN", ""),
    )

