"""
High-Speed Pure-Python & PyArrow HuggingFace Dataset Loader.
============================================================
Loads public HuggingFace datasets via:
  1. Native `.jsonl`, `.json`, and `.csv` files in the `main` branch (100% pure Python,
     zero binary `.pyd` DLL dependencies — works even under Windows Smart App Control).
  2. `pyarrow.parquet` when running on Linux / Kaggle GPU.
  3. HuggingFace `datasets-server` REST `/rows` pagination fallback when a repo only
     stores `.parquet` and `pyarrow` is blocked by Windows policy.
"""

import csv
import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional
from huggingface_hub import HfApi, hf_hub_download


def _fetch_rows_via_datasets_server(
    repo_id: str,
    token: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fallback loader using HuggingFace datasets-server `/rows` API in 100-row pages."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    records: List[Dict[str, Any]] = []
    try:
        sp_url = f"https://datasets-server.huggingface.co/splits?dataset={urllib.parse.quote(repo_id)}"
        req = urllib.request.Request(sp_url, headers=headers)
        splits = json.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8")).get("splits", [])
    except Exception as exc:
        print(f"  [WARN] datasets-server splits error for {repo_id}: {exc}")
        return records

    for sp_info in splits:
        cfg = sp_info["config"]
        sp = sp_info["split"]
        offset = 0
        page_size = 100
        while True:
            if max_rows and len(records) >= max_rows:
                return records[:max_rows]
            row_url = (
                f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(repo_id)}"
                f"&config={urllib.parse.quote(cfg)}&split={urllib.parse.quote(sp)}"
                f"&offset={offset}&length={page_size}"
            )
            try:
                r_req = urllib.request.Request(row_url, headers=headers)
                data = json.loads(urllib.request.urlopen(r_req, timeout=20).read().decode("utf-8"))
                batch = data.get("rows", [])
                if not batch:
                    break
                for item in batch:
                    row_dict = item.get("row", {})
                    # Drop heavy media blobs if present
                    row_dict.pop("video", None)
                    row_dict.pop("image", None)
                    records.append(row_dict)
                if len(batch) < page_size:
                    break
                offset += page_size
            except Exception as exc:
                print(f"  [WARN] datasets-server rows page error ({repo_id} offset={offset}): {exc}")
                break
    return records[:max_rows] if max_rows else records


def load_hf_dataset_rows(
    repo_id: str,
    token: Optional[str] = None,
    cache_dir: str = "data/cache",
    max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Download and parse all rows from a HuggingFace dataset repository.
    Prioritizes native `.jsonl` and `.csv` files first for maximum speed and
    cross-platform compatibility, then tries `pyarrow.parquet` or `datasets-server`.
    """
    api = HfApi(token=token)
    os.makedirs(cache_dir, exist_ok=True)
    records: List[Dict[str, Any]] = []

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision="main")
    except Exception as exc:
        print(f"  [WARN] Could not list main branch for {repo_id}: {exc}")
        files = []

    jsonl_files = [f for f in files if f.endswith(".jsonl")]
    # If repo has a top-level cases.jsonl or data/train.jsonl, avoid double-loading split copies
    if "cases.jsonl" in jsonl_files:
        jsonl_files = ["cases.jsonl"]
    elif "manim_narrated_400.jsonl" in jsonl_files and "train.jsonl" in jsonl_files:
        jsonl_files = ["train.jsonl"]

    csv_files = [f for f in files if f.endswith(".csv")]
    parquet_files = [f for f in files if f.endswith(".parquet")]

    # 1. Native JSONL files (fastest, 100% pure Python)
    if jsonl_files:
        for jf in jsonl_files:
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=jf,
                    repo_type="dataset",
                    revision="main",
                    token=token,
                    cache_dir=cache_dir,
                )
                with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
                            if max_rows and len(records) >= max_rows:
                                return records[:max_rows]
            except Exception as exc:
                print(f"  [WARN] Error reading jsonl {jf} from {repo_id}: {exc}")
        if records:
            return records[:max_rows] if max_rows else records

    # 2. Native CSV files (100% pure Python)
    if csv_files:
        for cf in csv_files:
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=cf,
                    repo_type="dataset",
                    revision="main",
                    token=token,
                    cache_dir=cache_dir,
                )
                csv.field_size_limit(10_000_000)
                with open(local_path, "r", encoding="utf-8", errors="replace", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        records.append(dict(row))
                        if max_rows and len(records) >= max_rows:
                            return records[:max_rows]
            except Exception as exc:
                print(f"  [WARN] Error reading csv {cf} from {repo_id}: {exc}")
        if records:
            return records[:max_rows] if max_rows else records

    # 3. Parquet files (Try PyArrow first; if blocked by Windows AppControl, use datasets-server REST API)
    if parquet_files:
        try:
            import pyarrow.parquet as pq  # type: ignore

            for pf in parquet_files:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=pf,
                    repo_type="dataset",
                    revision="main",
                    token=token,
                    cache_dir=cache_dir,
                )
                schema = pq.read_schema(local_path)
                cols_to_read = [
                    name for name in schema.names
                    if name not in {"video", "image", "audio", "result_files"}
                ]
                table = pq.read_table(local_path, columns=cols_to_read)
                records.extend(table.to_pylist())
                if max_rows and len(records) >= max_rows:
                    return records[:max_rows]
            if records:
                return records[:max_rows] if max_rows else records
        except Exception:
            pass

    # 4. Fallback to datasets-server REST pagination
    return _fetch_rows_via_datasets_server(repo_id=repo_id, token=token, max_rows=max_rows)
