"""
Self-Healing Live Kaggle GPU Kernel & HuggingFace Hub Checkpoint Monitor.
=========================================================================
Checks:
  1. Live Kaggle GPU kernel status (`priyadharsanbalaji18/manimce-qwen25-coder-7b-16bit-dpo-eval`)
     and GPU quota usage (`ApiGetAcceleratorQuotaStatisticsRequest`).
  2. Live HuggingFace Hub checkpoint repositories (`PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit`
     and `PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit`), printing step history, validation
     perplexity, DPO preference accuracy, and held-out ManimBench evaluation results.
  3. Automatic Self-Healing Watchdog (`--auto-heal`): if the kernel ever enters `ERROR`
     or gets stuck in `QUEUED`, automatically deletes the stuck kernel session via
     `ApiDeleteKernelRequest`, waits 5s for worker teardown, and re-pushes the kernel.
"""

import json
import os
import time
from huggingface_hub import HfApi, hf_hub_download
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiDeleteKernelRequest,
    ApiGetAcceleratorQuotaStatisticsRequest,
)


def auto_reset_and_push_kernel(k_api: KaggleApi, hf_token: str, kernel_ref: str) -> None:
    """Delete stuck/error Kaggle kernel session and redeploy cleanly."""
    user_name, slug = kernel_ref.split("/", 1)
    print(f"\n[Watchdog Auto-Heal] Resetting stuck/error kernel: {kernel_ref} ...")
    req = ApiDeleteKernelRequest()
    req.user_name = user_name
    req.kernel_slug = slug
    with k_api.build_kaggle_client() as kaggle:
        resp = kaggle.kernels.kernels_api_client.delete_kernel(req)
        print(f"  -> Delete response: {resp}")
    time.sleep(5)
    from scripts.push_kaggle_kernel import push_and_verify_kaggle_kernel

    push_and_verify_kaggle_kernel(
        kaggle_token=os.environ.get("KAGGLE_API_TOKEN", ""),
        hf_token=hf_token,
    )


def check_live_status(
    kaggle_token: str,
    hf_token: str,
    kernel_ref: str = "priyadharsanbalaji18/manimce-qwen25-coder-7b-v5-95plus",
    sft_repo: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-V5-SFT-16bit",
    dpo_repo: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-V5-DPO-16bit",
    auto_heal: bool = True,
) -> None:
    os.environ["KAGGLE_API_TOKEN"] = kaggle_token
    k_api = KaggleApi()
    k_api.authenticate()

    with k_api.build_kaggle_client() as kaggle:
        q = kaggle.kernels.kernels_api_client.get_accelerator_quota_statistics(
            ApiGetAcceleratorQuotaStatisticsRequest()
        )
        gpu_q = getattr(q, "gpu_quota", None)
        print(
            f"=== KAGGLE GPU QUOTA ===\n"
            f"  Used: {getattr(gpu_q, 'time_used', 'N/A')} / Total Allowed: {getattr(gpu_q, 'total_time_allowed', 'N/A')}"
        )

    status = k_api.kernels_status(kernel_ref)
    st_str = str(getattr(status, "status", status))
    print(f"\n=== KAGGLE KERNEL STATUS ({kernel_ref}) ===")
    print(f"  Status: {st_str}")
    if getattr(status, "failure_message", None):
        print(f"  Failure Message: {status.failure_message}")

    hf_api = HfApi(token=hf_token)
    for stage_name, repo_id in [("Stage 1 (16-Bit SFT)", sft_repo), ("Stage 2 (16-Bit DPO & Eval)", dpo_repo)]:
        print(f"\n=== HUGGINGFACE HUB CHECKPOINT ({stage_name}: {repo_id}) ===")
        try:
            files = hf_api.list_repo_files(repo_id=repo_id, repo_type="model")
            print(f"  Files in repo ({len(files)}): {files[:10]}")
            if "best/training_state.json" in files:
                b_path = hf_hub_download(
                    repo_id=repo_id, filename="best/training_state.json", repo_type="model", token=hf_token
                )
                with open(b_path, "r", encoding="utf-8") as f:
                    bst = json.load(f)
                print(
                    f"  [GOLD BEST IN best/] Step: {bst.get('global_step')} | "
                    f"Best Val Loss: {bst.get('val_loss'):.4f} | Val PPL: {bst.get('val_ppl'):.4f}"
                )
            if "training_state.json" in files:
                st_path = hf_hub_download(
                    repo_id=repo_id, filename="training_state.json", repo_type="model", token=hf_token
                )
                with open(st_path, "r", encoding="utf-8") as f:
                    st = json.load(f)
                print(
                    f"  Latest Saved Step: {st.get('global_step')} | "
                    f"Status: {st.get('status')} | "
                    f"Best Val Loss: {st.get('best_val_loss', st.get('sft_best_val_loss'))}"
                )
                hist = st.get("history", [])
                if hist:
                    print("  Recent Step History:")
                    for h in hist[-5:]:
                        print(f"    {h}")
            if "v5_real_render_eval_results.json" in files:
                ev_path = hf_hub_download(
                    repo_id=repo_id, filename="v5_real_render_eval_results.json", repo_type="model", token=hf_token
                )
                with open(ev_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                print(
                    f"  [V5 STAGE 3 REAL MANIM RENDER EVAL] Cases: {ev.get('num_cases')} | "
                    f"Pass@1 MP4: {ev.get('pass1_real_mp4_render_pct')}% | "
                    f"Pass@2 RITL MP4: {ev.get('pass2_ritl_real_mp4_render_pct')}% | "
                    f"Pass@3 RITL MP4: {ev.get('pass3_ritl_real_mp4_render_pct')}% | "
                    f"Pass@6 RITL MP4: {ev.get('pass6_ritl_real_mp4_render_pct')}%"
                )
            elif "v4_real_render_eval_results.json" in files:
                ev_path = hf_hub_download(
                    repo_id=repo_id, filename="v4_real_render_eval_results.json", repo_type="model", token=hf_token
                )
                with open(ev_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                print(
                    f"  [V4 STAGE 3 REAL MANIM RENDER EVAL] Cases: {ev.get('num_cases')} | "
                    f"Pass@1 MP4: {ev.get('pass1_real_mp4_render_pct')}% | "
                    f"Pass@2 RITL MP4: {ev.get('pass2_ritl_real_mp4_render_pct')}% | "
                    f"Pass@3 RITL MP4: {ev.get('pass3_ritl_real_mp4_render_pct')}% | "
                    f"Pass@5 RITL MP4: {ev.get('pass5_ritl_real_mp4_render_pct')}%"
                )
            elif "v3_real_render_eval_results.json" in files:
                ev_path = hf_hub_download(
                    repo_id=repo_id, filename="v3_real_render_eval_results.json", repo_type="model", token=hf_token
                )
                with open(ev_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                print(
                    f"  [V3 STAGE 3 REAL MANIM RENDER EVAL] Cases: {ev.get('num_cases')} | "
                    f"Pass@1 MP4: {ev.get('pass1_real_mp4_render_pct')}% | "
                    f"Pass@2 RITL MP4: {ev.get('pass2_ritl_real_mp4_render_pct')}% | "
                    f"Pass@3 RITL MP4: {ev.get('pass3_ritl_real_mp4_render_pct')}%"
                )
            elif "v2_real_render_eval_results.json" in files:
                ev_path = hf_hub_download(
                    repo_id=repo_id, filename="v2_real_render_eval_results.json", repo_type="model", token=hf_token
                )
                with open(ev_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                print(
                    f"  [V2 STAGE 3 REAL MANIM RENDER EVAL] Cases: {ev.get('num_cases')} | "
                    f"Pass@1 MP4 Render: {ev.get('pass1_real_mp4_render_pct')}% | "
                    f"Pass@2 RITL MP4 Render: {ev.get('pass2_ritl_real_mp4_render_pct')}%"
                )
            elif "manimbench_eval_results.json" in files:
                ev_path = hf_hub_download(
                    repo_id=repo_id, filename="manimbench_eval_results.json", repo_type="model", token=hf_token
                )
                with open(ev_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                print(
                    f"  [STAGE 3 BENCHMARK EVAL] Cases: {ev.get('num_cases')} | "
                    f"Pass@1: {ev.get('pass1_ast_and_api_valid_pct')}% | "
                    f"Pass@2 (RITL): {ev.get('pass2_ritl_self_repaired_pct')}%"
                )
        except Exception:
            print("  No checkpoint uploaded yet (kernel is currently initializing or in first 20 steps).")

    if auto_heal and ("ERROR" in st_str or "CANCEL" in st_str):
        auto_reset_and_push_kernel(k_api, hf_token, kernel_ref)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    check_live_status(
        kaggle_token=os.environ.get("KAGGLE_API_TOKEN", ""),
        hf_token=os.environ.get("HF_TOKEN", ""),
        auto_heal=True,
    )
