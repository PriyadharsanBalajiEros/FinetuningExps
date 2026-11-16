"""
Master ManimCE v0.19+ Dataset Curation, AST Sanitization & Hub Upload Pipeline.
===============================================================================
Downloads all major public Manim datasets from HuggingFace, runs:
  1. Static AST verification & ManimCE v0.19+ canonicalization (`manim_canonicalizer.py`)
  2. Prompt-to-Code semantic/lexical alignment filtering (`alignment_filter.py`)
  3. Exact & near-duplicate code hash deduplication (with zero eval-set leakage)
  4. Multi-turn Renderer-in-the-Loop (RITL) compiler repair synthesis (`dpo_pair_builder.py`)
  5. Hard-negative DPO preference pair construction (`dpo_pair_builder.py`)
and uploads the unified `ManimCE-Master-SFT` and `ManimCE-Master-DPO` datasets
to HuggingFace Hub (`PRB1`).
"""

import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from huggingface_hub import HfApi

# Allow running directly from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.alignment_filter import verify_prompt_code_alignment
from src.data_pipeline.dpo_pair_builder import (
    SYSTEM_PROMPT_MANIMCE,
    build_dpo_and_ritl_records,
)
from src.data_pipeline.hf_loader import load_hf_dataset_rows
from src.data_pipeline.manim_canonicalizer import (
    extract_python_code,
    verify_and_canonicalize,
)


def _save_jsonl(records: List[Dict[str, Any]], filepath: str) -> None:
    """Write a list of dictionaries to a UTF-8 JSONL file (100% pure Python)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _normalize_code_hash(code: str) -> str:
    """Compute a whitespace-invariant hash of Python code for strict deduplication."""
    stripped = re.sub(r"#.*", "", code)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return hashlib.sha256(stripped.encode("utf-8", errors="ignore")).hexdigest()


def _clean_user_prompt(raw_prompt: str) -> str:
    """Strip boilerplate wrapper prefixes from synthetic instruction datasets."""
    p = (raw_prompt or "").strip()
    prefixes = [
        "Generate accurate and correct ManimCE Python code for the animation requested by the user. Here is the user's request:",
        "Generate accurate and correct ManimCE Python code for the animation requested by the user.",
    ]
    for pref in prefixes:
        if p.startswith(pref):
            p = p[len(pref):].strip()
    return p


def _is_mostly_ascii_or_math(text: str) -> bool:
    """Keep English + mathematical LaTeX prompts so tokenization stays consistent."""
    if not text:
        return False
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    return cjk_chars <= 5


def _extract_messages_prompt_and_code(messages: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract (user_prompt, assistant_code_or_response, reasoning_trace) from messages list."""
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return None, None, None
    if isinstance(messages, dict) and "messages" in messages:
        messages = messages["messages"]
    if not isinstance(messages, list):
        return None, None, None

    user_prompt = None
    assistant_reply = None
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("from", "")
        content = m.get("content") or m.get("value", "")
        if role in {"user", "human"} and not user_prompt:
            user_prompt = str(content).strip()
        elif role in {"assistant", "gpt", "model"} and content:
            assistant_reply = str(content).strip()

    if not user_prompt or not assistant_reply:
        return None, None, None

    think_match = re.search(r"<think>(.*?)</think>", assistant_reply, flags=re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else None
    return user_prompt, assistant_reply, reasoning


def curate_all_datasets(
    hf_token: str,
    output_dir: str = "data/processed",
    cache_dir: str = "data/cache",
) -> Dict[str, Any]:
    """Run the full multi-dataset ingestion, verification, alignment, and split pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    audit_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "raw_rows": 0,
            "ast_valid": 0,
            "aligned_kept": 0,
            "dedup_kept": 0,
            "repairs_count": 0,
            "rejection_reasons": Counter(),
        }
    )

    seen_code_hashes: Set[str] = set()
    eval_benchmark_records: List[Dict[str, Any]] = []
    sft_records: List[Dict[str, Any]] = []
    external_dpo_records: List[Dict[str, Any]] = []

    def register_record(
        source_name: str,
        prompt: str,
        raw_code: str,
        reasoning_trace: Optional[str] = None,
        tier: int = 1,
        is_trusted: bool = True,
        is_eval_split: bool = False,
    ) -> bool:
        stats = audit_stats[source_name]
        stats["raw_rows"] += 1

        prompt_clean = _clean_user_prompt(prompt)
        if len(prompt_clean) < 12 or not _is_mostly_ascii_or_math(prompt_clean):
            stats["rejection_reasons"]["invalid_or_non_english_prompt"] += 1
            return False

        vres = verify_and_canonicalize(raw_code)
        if not vres.is_valid:
            reason_key = (vres.rejection_reason or "invalid_code").split(":")[0]
            stats["rejection_reasons"][reason_key] += 1
            return False
        stats["ast_valid"] += 1
        if vres.repairs_applied:
            stats["repairs_count"] += 1

        ares = verify_prompt_code_alignment(
            prompt_clean, vres.canonical_code, is_trusted_source=is_trusted
        )
        if not ares.is_aligned:
            stats["rejection_reasons"][ares.reason.split(":")[0]] += 1
            return False
        stats["aligned_kept"] += 1

        code_hash = _normalize_code_hash(vres.canonical_code)
        if code_hash in seen_code_hashes:
            stats["rejection_reasons"]["duplicate_code_hash"] += 1
            return False
        seen_code_hashes.add(code_hash)
        stats["dedup_kept"] += 1

        # Format assistant completion (with optional <think> spatial/math planning trace)
        if reasoning_trace and _is_mostly_ascii_or_math(reasoning_trace) and len(reasoning_trace.strip()) > 30:
            clean_trace = reasoning_trace.strip()
            assistant_content = f"<think>\n{clean_trace}\n</think>\n\n```python\n{vres.canonical_code}\n```"
            has_reasoning = True
        else:
            assistant_content = f"```python\n{vres.canonical_code}\n```"
            has_reasoning = False

        record = {
            "id": f"{source_name.split('/')[-1]}_{stats['dedup_kept']:06d}",
            "source": source_name,
            "tier": tier,
            "scene_name": vres.scene_name,
            "scene_base": vres.scene_base,
            "num_animations": vres.num_animations,
            "num_lines": vres.num_lines,
            "has_reasoning": has_reasoning,
            "prompt": prompt_clean,
            "code": vres.canonical_code,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_MANIMCE},
                {"role": "user", "content": prompt_clean},
                {"role": "assistant", "content": assistant_content},
            ],
        }

        if is_eval_split:
            eval_benchmark_records.append(record)
        else:
            sft_records.append(record)
        return True

    # =========================================================================
    # 1. HELD-OUT BENCHMARK & GOLD RENDER-VERIFIED DATASETS (Process First!)
    # =========================================================================
    print("\n[1/8] Processing Gold Render-Verified Dataset: sebastianboehler/autoresearch-manim ...")
    rows = load_hf_dataset_rows("sebastianboehler/autoresearch-manim", token=hf_token, cache_dir=cache_dir)
    for idx, r in enumerate(rows):
        prompt = r.get("prompt", "")
        code = r.get("completion", "")
        is_eval = (idx % 5 == 0)
        register_record(
            "sebastianboehler/autoresearch-manim",
            prompt,
            code,
            tier=1,
            is_trusted=True,
            is_eval_split=is_eval,
        )

    print("[2/8] Processing Benchmark Dataset: SuienR/ManimBench-v1 ...")
    rows = load_hf_dataset_rows("SuienR/ManimBench-v1", token=hf_token, cache_dir=cache_dir)
    for idx, r in enumerate(rows):
        prompt = r.get("Reviewed Description") or r.get("Generated Description") or ""
        code = r.get("Code", "")
        split_val = str(r.get("Split", "train")).lower()
        is_eval = split_val in {"test", "val", "validation"} or (idx % 8 == 0 and len(eval_benchmark_records) < 160)
        register_record(
            "SuienR/ManimBench-v1",
            prompt,
            code,
            tier=1,
            is_trusted=True,
            is_eval_split=is_eval,
        )

    # =========================================================================
    # 2. BESPOKE-MANIM (Execution-Verified ManimCE v0.19.0 + Spatial Planning)
    # =========================================================================
    print("[3/8] Processing Execution-Verified Dataset: bespokelabs/bespoke-manim ...")
    rows = load_hf_dataset_rows("bespokelabs/bespoke-manim", token=hf_token, cache_dir=cache_dir)
    for r in rows:
        if r.get("error") is not None:
            audit_stats["bespokelabs/bespoke-manim"]["raw_rows"] += 1
            audit_stats["bespokelabs/bespoke-manim"]["rejection_reasons"]["runtime_error_in_source"] += 1
            continue
        question = r.get("question") or ""
        title = r.get("title") or ""
        topic = r.get("topic") or ""
        style = r.get("visual_style") or ""
        prompt = f"Create an educational ManimCE animation on '{topic}' ({title}): {question}"
        if style:
            prompt += f"\nVisual style guidance: {style}"

        vis_elems = r.get("visual_elements") or []
        eqs = r.get("equations") or []
        think_parts = [f"Topic: {topic} — {title}"]
        if isinstance(vis_elems, list) and vis_elems:
            steps_str = "; ".join(
                f"[{v.get('timestamp', '')}] {v.get('description', '')}"
                for v in vis_elems[:6] if isinstance(v, dict)
            )
            if steps_str:
                think_parts.append(f"Visual Timeline Plan: {steps_str}")
        if isinstance(eqs, list) and eqs:
            think_parts.append("Key Equations (MathTex): " + ", ".join(str(e) for e in eqs[:4]))
        reasoning = "\n".join(think_parts) if len(think_parts) > 1 else None

        register_record(
            "bespokelabs/bespoke-manim",
            prompt,
            r.get("python_code", ""),
            reasoning_trace=reasoning,
            tier=1,
            is_trusted=True,
        )

    # =========================================================================
    # 3. HIGH-QUALITY MODERN MANIMCE v0.19+ SFT DATASETS
    # =========================================================================
    chat_sft_sources = [
        ("nabin2004/qwen3-8b-manimator-gold-sft", "messages", 1, True),
        ("nabin2004/AOS-Manim-SFT", "messages", 1, True),
        ("nabin2004/AOS-Narrated-Manim-400", "messages", 1, True),
        ("bakhil-aissa/Manim_code_traces", "messages", 1, True),
        ("nabin2004/manim-sft-10k", "messages", 1, True),
        ("Chimanwakis/manim_animation_dataset", "conversations", 1, True),
    ]
    print("[4/8] Processing Modern Multi-Turn ManimCE SFT Datasets ...")
    for repo_id, col_name, tier, trusted in chat_sft_sources:
        print(f"  -> Loading {repo_id} ...")
        rows = load_hf_dataset_rows(repo_id, token=hf_token, cache_dir=cache_dir)
        for r in rows:
            u_prompt, a_reply, reasoning = _extract_messages_prompt_and_code(r.get(col_name))
            if u_prompt and a_reply:
                register_record(repo_id, u_prompt, a_reply, reasoning_trace=reasoning, tier=tier, is_trusted=trusted)

    # =========================================================================
    # 4. PAIRWISE PROMPT-CODE DATASETS (Trusted & Untrusted)
    # =========================================================================
    pair_sources = [
        ("RY7Games/ry_manim_codegen", "query", "output", 1, True),
        ("Noxus09/manim-dataset", "prompt", "Code", 1, True),
        ("Edoh/manim_python", "instruction", "output", 1, True),
        ("mistral-hackaton-2026/manim-finetune-data", "prompt", "code", 2, False),
    ]
    print("[5/8] Processing Pairwise Prompt-Code Datasets (with strict alignment filter) ...")
    for repo_id, p_col, c_col, tier, trusted in pair_sources:
        print(f"  -> Loading {repo_id} (trusted={trusted}) ...")
        rows = load_hf_dataset_rows(repo_id, token=hf_token, cache_dir=cache_dir)
        for r in rows:
            register_record(
                repo_id,
                str(r.get(p_col, "")),
                str(r.get(c_col, "")),
                tier=tier,
                is_trusted=trusted,
            )

    # =========================================================================
    # 5. ADAPTION-UNIFIED-MANIM-INSTRUCTIONS (CoT Reasoning + Enhanced Completion)
    # =========================================================================
    print("[6/8] Processing CoT Reasoning Dataset: priyanshutomar2024/adaption-unified-manim-instructions ...")
    rows = load_hf_dataset_rows(
        "priyanshutomar2024/adaption-unified-manim-instructions",
        token=hf_token,
        cache_dir=cache_dir,
    )
    for r in rows:
        raw_inst = str(r.get("instruction") or r.get("enhanced_prompt") or "")
        if "class " in raw_inst and "def construct" in raw_inst:
            audit_stats["priyanshutomar2024/adaption-unified-manim-instructions"]["raw_rows"] += 1
            audit_stats["priyanshutomar2024/adaption-unified-manim-instructions"]["rejection_reasons"]["code_leaked_in_prompt"] += 1
            continue
        enh_comp = str(r.get("enhanced_completion") or "")
        r_trace = str(r.get("reasoning_trace") or "")
        register_record(
            "priyanshutomar2024/adaption-unified-manim-instructions",
            raw_inst,
            enh_comp,
            reasoning_trace=r_trace,
            tier=2,
            is_trusted=False,
        )

    # =========================================================================
    # 6. DPO PREFERENCE PAIRS & RITL SELF-REPAIR TRAJECTORIES
    # =========================================================================
    print("[7/8] Processing DPO Preference Dataset & Synthesizing Hard Compiler Negatives ...")
    dpo_rows = load_hf_dataset_rows("nabin2004/manim-narrated-dpo-400", token=hf_token, cache_dir=cache_dir)
    for r in dpo_rows:
        prompt = _clean_user_prompt(str(r.get("prompt", "")))
        chosen_raw = str(r.get("chosen", ""))
        rejected_raw = str(r.get("rejected", ""))
        v_chosen = verify_and_canonicalize(chosen_raw)
        rejected_code = extract_python_code(rejected_raw)
        if v_chosen.is_valid and rejected_code and prompt:
            external_dpo_records.append(
                {
                    "system": SYSTEM_PROMPT_MANIMCE,
                    "prompt": prompt,
                    "chosen": f"```python\n{v_chosen.canonical_code}\n```",
                    "rejected": f"```python\n{rejected_code}\n```",
                    "category": "narrated_pedagogical_preference",
                    "source": "nabin2004/manim-narrated-dpo-400",
                }
            )

    synth_dpo, ritl_repairs = build_dpo_and_ritl_records(
        sft_records,
        max_synthetic_dpo=2400,
        max_ritl_repairs=1200,
    )
    all_dpo_records = external_dpo_records + synth_dpo
    sft_records.extend(ritl_repairs)

    # Shuffle deterministically and split SFT into train (97%) and val (3%)
    rng = random.Random(2026)
    rng.shuffle(sft_records)
    rng.shuffle(all_dpo_records)

    val_size = min(500, max(150, int(0.03 * len(sft_records))))
    sft_val = sft_records[:val_size]
    sft_train = sft_records[val_size:]

    dpo_val_size = min(150, max(50, int(0.05 * len(all_dpo_records))))
    dpo_val = all_dpo_records[:dpo_val_size]
    dpo_train = all_dpo_records[dpo_val_size:]

    # =========================================================================
    # 7. SAVE JSONL SPLITS & AUDIT REPORT LOCALLY
    # =========================================================================
    print("[8/8] Saving curated JSONL splits and audit report ...")
    _save_jsonl(sft_train, os.path.join(output_dir, "sft_train.jsonl"))
    _save_jsonl(sft_val, os.path.join(output_dir, "sft_val.jsonl"))
    _save_jsonl(eval_benchmark_records, os.path.join(output_dir, "eval_benchmark.jsonl"))
    _save_jsonl(dpo_train, os.path.join(output_dir, "dpo_train.jsonl"))
    _save_jsonl(dpo_val, os.path.join(output_dir, "dpo_val.jsonl"))

    summary = {
        "totals": {
            "sft_train": len(sft_train),
            "sft_val": len(sft_val),
            "sft_with_reasoning_traces": sum(1 for r in sft_train if r.get("has_reasoning")),
            "sft_ritl_self_repair": len(ritl_repairs),
            "dpo_train": len(dpo_train),
            "dpo_val": len(dpo_val),
            "eval_benchmark_heldout": len(eval_benchmark_records),
            "unique_code_hashes": len(seen_code_hashes),
        },
        "per_dataset_audit": {
            k: {
                "raw_rows": v["raw_rows"],
                "ast_valid": v["ast_valid"],
                "aligned_kept": v["aligned_kept"],
                "dedup_kept": v["dedup_kept"],
                "repairs_count": v["repairs_count"],
                "rejection_reasons": dict(v["rejection_reasons"]),
            }
            for k, v in audit_stats.items()
        },
    }

    report_path = os.path.join("data", "curation_audit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CURATION SUMMARY ===")
    print(json.dumps(summary["totals"], indent=2))
    return summary


def upload_curated_datasets_to_hub(
    hf_token: str,
    hf_username: str = "PRB1",
    output_dir: str = "data/processed",
) -> None:
    """Upload curated SFT and DPO JSONL datasets + Audit Report to HuggingFace Hub."""
    api = HfApi(token=hf_token)

    sft_repo = f"{hf_username}/ManimCE-Master-SFT"
    dpo_repo = f"{hf_username}/ManimCE-Master-DPO"

    print(f"\nUploading SFT Master Dataset to HuggingFace Hub: {sft_repo} ...")
    api.create_repo(repo_id=sft_repo, repo_type="dataset", exist_ok=True, private=False)
    for split_file, hub_path in [
        ("sft_train.jsonl", "data/train.jsonl"),
        ("sft_val.jsonl", "data/validation.jsonl"),
        ("eval_benchmark.jsonl", "data/test.jsonl"),
    ]:
        local_f = os.path.join(output_dir, split_file)
        if os.path.exists(local_f):
            api.upload_file(
                path_or_fileobj=local_f,
                path_in_repo=hub_path,
                repo_id=sft_repo,
                repo_type="dataset",
            )
    if os.path.exists("data/curation_audit_report.json"):
        api.upload_file(
            path_or_fileobj="data/curation_audit_report.json",
            path_in_repo="curation_audit_report.json",
            repo_id=sft_repo,
            repo_type="dataset",
        )

    print(f"Uploading DPO Master Dataset to HuggingFace Hub: {dpo_repo} ...")
    api.create_repo(repo_id=dpo_repo, repo_type="dataset", exist_ok=True, private=False)
    for split_file, hub_path in [
        ("dpo_train.jsonl", "data/train.jsonl"),
        ("dpo_val.jsonl", "data/validation.jsonl"),
    ]:
        local_f = os.path.join(output_dir, split_file)
        if os.path.exists(local_f):
            api.upload_file(
                path_or_fileobj=local_f,
                path_in_repo=hub_path,
                repo_id=dpo_repo,
                repo_type="dataset",
            )
    print("HuggingFace Hub dataset uploads complete!")


if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError("Please set HF_TOKEN environment variable.")
    curate_all_datasets(hf_token=token)
    upload_curated_datasets_to_hub(hf_token=token, hf_username="PRB1")
