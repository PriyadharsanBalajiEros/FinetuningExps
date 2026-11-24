"""
Automated Kaggle GPU Kernel Deployment & Status Monitor Script.
===============================================================
Prepares an isolated private build directory (`outputs/kaggle_build`),
injects the runtime HuggingFace token into the private script copy so
the remote Kaggle GPU worker can checkpoint directly to `PRB1` on HuggingFace
Hub every 100 steps, and pushes the kernel via `KaggleApi().kernels_push()`.
"""

import json
import os
import shutil
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def push_and_verify_kaggle_kernel(
    kaggle_token: str,
    hf_token: str,
    kernel_src_dir: str = "kaggle_kernel_v5",
    build_dir: str = "outputs/kaggle_build_v5",
) -> None:
    os.environ["KAGGLE_API_TOKEN"] = kaggle_token
    os.makedirs(build_dir, exist_ok=True)

    meta_src = os.path.join(kernel_src_dir, "kernel-metadata.json")
    with open(meta_src, "r", encoding="utf-8") as f:
        meta = json.load(f)
    code_file = meta["code_file"]
    script_src = os.path.join(kernel_src_dir, code_file)

    shutil.copyfile(meta_src, os.path.join(build_dir, "kernel-metadata.json"))
    with open(script_src, "r", encoding="utf-8") as f:
        code = f.read()

    # Inject HF token into the gitignored private build copy only
    code_injected = code.replace("__HF_TOKEN_PLACEHOLDER__", hf_token)
    with open(os.path.join(build_dir, code_file), "w", encoding="utf-8") as f:
        f.write(code_injected)

    kernel_ref = meta["id"]

    api = KaggleApi()
    api.authenticate()
    print(f"[Kaggle Deploy] Authenticated as '{api.config_values.get('username')}'. Pushing kernel: {kernel_ref} ...")
    resp = api.kernels_push(build_dir)
    print(f"[Kaggle Deploy] Push response: ref={getattr(resp, 'ref', kernel_ref)}, url={getattr(resp, 'url', '')}")

    time.sleep(4)
    status = api.kernels_status(kernel_ref)
    print(f"[Kaggle Status] {kernel_ref} -> status: {getattr(status, 'status', status)}")


if __name__ == "__main__":
    k_tok = os.environ.get("KAGGLE_API_TOKEN", "")
    h_tok = os.environ.get("HF_TOKEN", "")
    if not k_tok or not h_tok:
        raise ValueError("Both KAGGLE_API_TOKEN and HF_TOKEN environment variables are required.")
    push_and_verify_kaggle_kernel(kaggle_token=k_tok, hf_token=h_tok)

