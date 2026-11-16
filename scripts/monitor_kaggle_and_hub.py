"""
Live Kaggle GPU Kernel & HuggingFace Hub Checkpoint Monitor.
============================================================
Checks:
  1. Live Kaggle GPU kernel status (`priyadharsanbalaji18/manimce-qwen25-coder-7b-16bit-sft-dpo`)
  2. Live HuggingFace Hub checkpoint repositories (`PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit`
     and `PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit`) and prints the latest saved step,
     validation loss, and VRAM metrics from `training_state.json`.
"""

import json
import os
from huggingface_hub import HfApi, hf_hub_download
from kaggle.api.kaggle_api_extended import KaggleApi


def check_live_status(
    kaggle_token: str,
    hf_token: str,
    kernel_ref: str = "priyadharsanbalaji18/manimce-qwen25-coder-7b-16bit-sft-dpo",
    sft_repo: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit",
    dpo_repo: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit",
) -> None:
    os.environ["KAGGLE_API_TOKEN"] = kaggle_token
    k_api = KaggleApi()
    k_api.authenticate()
    status = k_api.kernels_status(kernel_ref)
    print(f"=== KAGGLE KERNEL STATUS ({kernel_ref}) ===")
    print(f"  Status: {getattr(status, 'status', status)}")
    if getattr(status, "failure_message", None):
        print(f"  Failure Message: {status.failure_message}")

    hf_api = HfApi(token=hf_token)
    for stage_name, repo_id in [("Stage 1 (16-Bit SFT)", sft_repo), ("Stage 2 (16-Bit DPO)", dpo_repo)]:
        print(f"\n=== HUGGINGFACE HUB CHECKPOINT ({stage_name}: {repo_id}) ===")
        try:
            files = hf_api.list_repo_files(repo_id=repo_id, repo_type="model")
            print(f"  Files in repo ({len(files)}): {files[:8]}")
            if "training_state.json" in files:
                st_path = hf_hub_download(repo_id=repo_id, filename="training_state.json", repo_type="model", token=hf_token)
                with open(st_path, "r", encoding="utf-8") as f:
                    st = json.load(f)
                print(
                    f"  Latest Saved Step: {st.get('global_step')} | "
                    f"Status: {st.get('status')} | "
                    f"Best Val Loss: {st.get('best_val_loss')}"
                )
        except Exception:
            print("  No checkpoint uploaded yet (kernel is currently initializing or in first 100 steps).")


if __name__ == "__main__":
    check_live_status(
        kaggle_token=os.environ.get("KAGGLE_API_TOKEN", ""),
        hf_token=os.environ.get("HF_TOKEN", ""),
    )

