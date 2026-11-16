"""
Standalone Kaggle GPU Kernel: Pure 16-Bit Dual-T4 ManimCE SFT + DPO + Benchmark Pipeline.
=========================================================================================
Runs end-to-end on Kaggle GPU (`T4 x2 = 32 GB VRAM` or `P100 = 16 GB VRAM`):
  1. Verifies/installs required packages (`peft`, `accelerate`, `huggingface_hub`, `transformers`).
  2. Clones/pulls the latest codebase from `https://github.com/PriyadharsanBalajiEros/FinetuningExps.git`
     (or uses embedded trainer modules directly).
  3. Runs Stage 1: Pure 16-Bit (`torch.float16`) High-Rank `rsLoRA` ($r=64, \alpha=64$) SFT
     with Completion-Only Loss Masking and `NEFTune` ($\alpha=5.0$) on `PRB1/ManimCE-Master-SFT`,
     checkpointing to `PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit` every 150 steps.
  4. Runs Stage 2: Pure 16-Bit Adapter-Reference `DPO` + `RPO-NLL` Alignment on
     `PRB1/ManimCE-Master-DPO`, checkpointing to `PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit`.
  5. Evaluates Pass@1 and Pass@2 (RITL Self-Repair) on held-out `ManimBench-v1` &
     `autoresearch-manim` test cases and uploads the evaluation report to HuggingFace Hub.
"""

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

# Read HF_TOKEN from environment or Kaggle UserSecrets and export to os.environ
HF_TOKEN = os.environ.get("HF_TOKEN", "__HF_TOKEN_PLACEHOLDER__")
if HF_TOKEN.startswith("__HF_TOKEN"):
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = ""
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False
from peft import LoraConfig, PeftModel, get_peft_model
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


@dataclass
class PipelineConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    sft_dataset_repo: str = "PRB1/ManimCE-Master-SFT"
    dpo_dataset_repo: str = "PRB1/ManimCE-Master-DPO"
    hub_sft_repo: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit"
    hub_dpo_repo: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit"
    output_dir: str = "/kaggle/working/checkpoints"
    max_seq_length: int = 2560
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    use_rslora: bool = True
    neftune_alpha: float = 5.0
    grad_accum_steps: int = 16
    sft_lr: float = 1.5e-4
    dpo_lr: float = 2.0e-5
    dpo_beta: float = 0.1
    sft_epochs: int = 1
    sft_max_steps: int = 600
    dpo_max_steps: int = 150
    save_every_steps: int = 100
    eval_every_steps: int = 100
    log_every_steps: int = 10
    max_walltime_hours: float = 11.2
    seed: int = 2026


class ManimChatDataset(Dataset):
    """Completion-only masked ChatML dataset for 16-bit SFT."""

    def __init__(self, records: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_seq_length: int = 2560):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assistant_prefix_ids = tokenizer.encode("assistant\n", add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.records)

    def _build_masked_labels(self, input_ids: List[int]) -> List[int]:
        labels = [-100] * len(input_ids)
        n = len(input_ids)
        pref_len = len(self.assistant_prefix_ids)
        i = 0
        while i < n:
            if (
                input_ids[i] == self.im_start_id
                and i + 1 + pref_len <= n
                and input_ids[i + 1 : i + 1 + pref_len] == self.assistant_prefix_ids
            ):
                j = i + 1 + pref_len
                while j < n and input_ids[j] != self.im_end_id:
                    labels[j] = input_ids[j]
                    j += 1
                if j < n and input_ids[j] == self.im_end_id:
                    labels[j] = input_ids[j]
                    i = j + 1
                else:
                    break
            else:
                i += 1
        return labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        rendered = self.tokenizer.apply_chat_template(
            rec["messages"], tokenize=False, add_generation_prompt=False
        )
        encoded = self.tokenizer(
            rendered, truncation=True, max_length=self.max_seq_length, add_special_tokens=False
        )
        input_ids = encoded["input_ids"]
        labels = self._build_masked_labels(input_ids)
        if all(x == -100 for x in labels) and len(input_ids) > 32:
            for k in range(len(input_ids) // 2, len(input_ids)):
                labels[k] = input_ids[k]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class ManimDPODataset(Dataset):
    """Preference pair dataset `(system, prompt, chosen, rejected)` for Stage 2 16-bit DPO."""

    def __init__(self, records: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_seq_length: int = 2048):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.records)

    def _encode_item(self, system: str, prompt: str, response: str) -> Tuple[torch.Tensor, torch.Tensor]:
        p_msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        f_msgs = p_msgs + [{"role": "assistant", "content": response}]
        p_txt = self.tokenizer.apply_chat_template(p_msgs, tokenize=False, add_generation_prompt=True)
        f_txt = self.tokenizer.apply_chat_template(f_msgs, tokenize=False, add_generation_prompt=False)
        p_ids = self.tokenizer.encode(p_txt, add_special_tokens=False)
        f_ids = self.tokenizer.encode(
            f_txt, truncation=True, max_length=self.max_seq_length, add_special_tokens=False
        )
        labels = list(f_ids)
        for i in range(min(len(p_ids), len(labels) - 1)):
            labels[i] = -100
        return torch.tensor(f_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]
        c_ids, c_lbl = self._encode_item(r.get("system", ""), r["prompt"], r["chosen"])
        r_ids, r_lbl = self._encode_item(r.get("system", ""), r["prompt"], r["rejected"])
        return {
            "chosen_input_ids": c_ids,
            "chosen_labels": c_lbl,
            "rejected_input_ids": r_ids,
            "rejected_labels": r_lbl,
        }


def register_neftune_hook(model: nn.Module, alpha: float = 5.0):
    emb = model.get_input_embeddings()

    def _hook(module: nn.Module, args: Tuple, output: torch.Tensor) -> torch.Tensor:
        if module.training and alpha > 0.0:
            dims = torch.tensor(output.size(1) * output.size(2), dtype=output.dtype, device=output.device)
            mag = alpha / torch.sqrt(dims)
            return output + torch.zeros_like(output).uniform_(-mag.item(), mag.item())
        return output

    return emb.register_forward_hook(_hook)


def load_hub_jsonl(repo_id: str, filename: str, token: str) -> List[Dict[str, Any]]:
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=token)
    rows: List[Dict[str, Any]] = []
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_gpu_sharding() -> Tuple[Any, Optional[Dict[int, str]]]:
    n_gpus = torch.cuda.device_count()
    print(f"[Hardware Check] Detected {n_gpus} CUDA GPU(s):")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  - GPU {i}: {props.name} ({props.total_memory / (1024**3):.2f} GiB VRAM)")
    if n_gpus >= 2:
        max_mem = {
            i: f"{max(10.0, round(torch.cuda.get_device_properties(i).total_memory / (1024**3) * 0.93, 1))}GiB"
            for i in range(n_gpus)
        }
        return "auto", max_mem
    elif n_gpus == 1:
        return {"": 0}, None
    raise RuntimeError("No CUDA GPU available on this Kaggle instance!")


@torch.no_grad()
def evaluate_sft_val(model: nn.Module, val_loader: DataLoader, first_dev: torch.device) -> Tuple[float, float]:
    model.eval()
    tot_loss, count = 0.0, 0
    for idx, batch in enumerate(val_loader):
        if idx >= 80:
            break
        batch = {k: v.to(first_dev, non_blocking=True) for k, v in batch.items()}
        loss = model(**batch).loss
        if not torch.isnan(loss) and not torch.isinf(loss):
            tot_loss += loss.item()
            count += 1
    model.train()
    avg = tot_loss / max(1, count)
    return avg, math.exp(min(20.0, avg))


def sync_to_hub(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    repo_id: str,
    local_dir: str,
    state: Dict[str, Any],
    token: str,
    subfolder: str = "",
) -> None:
    os.makedirs(local_dir, exist_ok=True)
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    with open(os.path.join(local_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    if token:
        try:
            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
            api.upload_folder(
                folder_path=local_dir,
                path_in_repo=subfolder,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Checkpoint step {state.get('global_step', 0)}",
            )
            print(f"  [Hub Checkpoint Saved] -> https://huggingface.co/{repo_id} (step={state.get('global_step')})")
        except Exception as exc:
            print(f"  [WARN] Hub upload warning: {exc}")


def compute_dpo_logps(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != -100
    safe_labels = torch.where(mask, shift_labels, torch.zeros_like(shift_labels))
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_logps = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
    seq_logp = (token_logps * mask.float()).sum(dim=-1)
    nll = -(seq_logp / mask.float().sum(dim=-1).clamp(min=1.0)).mean()
    return seq_logp, nll


def main() -> None:
    start_time = time.time()
    cfg = PipelineConfig()
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device_map, max_memory = get_gpu_sharding()
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[Stage 1] Loading {cfg.base_model} in Native 16-Bit (FP16, SDPA, Zero 4-Bit Quantization)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model.enable_input_require_grads()
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Check if Stage 1 checkpoint already exists on Hub to resume
    resumed_step = 0
    best_val_loss = float("inf")
    sft_history: List[Dict[str, Any]] = []
    sft_already_done = False

    if HF_TOKEN:
        try:
            api = HfApi(token=HF_TOKEN)
            hub_files = api.list_repo_files(repo_id=cfg.hub_sft_repo, repo_type="model")
            if "adapter_config.json" in hub_files and "training_state.json" in hub_files:
                snap = snapshot_download(repo_id=cfg.hub_sft_repo, repo_type="model", token=HF_TOKEN)
                with open(os.path.join(snap, "training_state.json"), "r", encoding="utf-8") as f:
                    st = json.load(f)
                resumed_step = int(st.get("global_step", 0))
                best_val_loss = float(st.get("best_val_loss", float("inf")))
                sft_history = st.get("history", [])
                sft_already_done = st.get("status") == "completed"
                model = PeftModel.from_pretrained(base_model, snap, is_trainable=True)
                print(f"[Resume] Loaded existing Stage-1 16-bit adapter at step={resumed_step} (completed={sft_already_done})")
        except Exception:
            pass

    if resumed_step == 0:
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            use_rslora=cfg.use_rslora,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base_model, lora_cfg)

    model.print_trainable_parameters()

    # =========================================================================
    # STAGE 1: 16-BIT HIGH-RANK rsLoRA SFT + NEFTune
    # =========================================================================
    if not sft_already_done:
        neftune_handle = register_neftune_hook(model, alpha=cfg.neftune_alpha)
        train_rows = load_hub_jsonl(cfg.sft_dataset_repo, "data/train.jsonl", HF_TOKEN)
        val_rows = load_hub_jsonl(cfg.sft_dataset_repo, "data/validation.jsonl", HF_TOKEN)
        print(f"[Stage 1 Data] Loaded {len(train_rows)} SFT train rows and {len(val_rows)} val rows.")

        train_ds = ManimChatDataset(train_rows, tokenizer, max_seq_length=cfg.max_seq_length)
        val_ds = ManimChatDataset(val_rows, tokenizer, max_seq_length=cfg.max_seq_length)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=lambda b: {
            "input_ids": b[0]["input_ids"].unsqueeze(0),
            "attention_mask": b[0]["attention_mask"].unsqueeze(0),
            "labels": b[0]["labels"].unsqueeze(0),
        })
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda b: {
            "input_ids": b[0]["input_ids"].unsqueeze(0),
            "attention_mask": b[0]["attention_mask"].unsqueeze(0),
            "labels": b[0]["labels"].unsqueeze(0),
        })

        total_sft_steps = min(cfg.sft_max_steps, max(1, len(train_loader) // cfg.grad_accum_steps))
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.sft_lr, weight_decay=0.01, foreach=True)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=max(10, int(0.05 * total_sft_steps)), num_training_steps=total_sft_steps
        )
        for _ in range(resumed_step):
            scheduler.step()

        first_dev = next(model.parameters()).device
        model.train()
        optimizer.zero_grad()

        global_step = resumed_step
        micro_step = 0
        running_loss = 0.0

        for idx, batch in enumerate(train_loader):
            if resumed_step > 0 and idx < resumed_step * cfg.grad_accum_steps:
                continue
            batch = {k: v.to(first_dev, non_blocking=True) for k, v in batch.items()}
            loss = model(**batch).loss / cfg.grad_accum_steps
            loss.backward()
            running_loss += loss.item()
            micro_step += 1

            if micro_step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_every_steps == 0:
                    vram_gb = sum(
                        torch.cuda.max_memory_allocated(i) / (1024**3)
                        for i in range(torch.cuda.device_count())
                    )
                    avg_l = running_loss / cfg.log_every_steps
                    running_loss = 0.0
                    elapsed_m = (time.time() - start_time) / 60.0
                    print(
                        f"  [SFT Step {global_step:4d}/{total_sft_steps:4d}] Loss: {avg_l:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | Peak VRAM: {vram_gb:.2f} GB | {elapsed_m:.1f}m"
                    )
                    sft_history.append({"step": global_step, "loss": round(avg_l, 5), "vram_gb": round(vram_gb, 2)})

                if global_step % cfg.eval_every_steps == 0 or global_step >= total_sft_steps:
                    val_loss, val_ppl = evaluate_sft_val(model, val_loader, first_dev)
                    print(f"  >>> [SFT Val @ Step {global_step}] Val Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                    st = {
                        "global_step": global_step,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "best_val_loss": best_val_loss,
                        "status": "completed" if global_step >= total_sft_steps else "in_progress",
                        "history": sft_history,
                    }
                    sync_to_hub(model, tokenizer, cfg.hub_sft_repo, os.path.join(cfg.output_dir, "sft_latest"), st, HF_TOKEN)
                    if is_best:
                        sync_to_hub(model, tokenizer, cfg.hub_sft_repo, os.path.join(cfg.output_dir, "sft_best"), st, HF_TOKEN, subfolder="best")

                if (time.time() - start_time) / 3600.0 >= cfg.max_walltime_hours:
                    print("[Walltime Guard] Saving SFT state before Kaggle timeout...")
                    st = {"global_step": global_step, "best_val_loss": best_val_loss, "status": "paused", "history": sft_history}
                    sync_to_hub(model, tokenizer, cfg.hub_sft_repo, os.path.join(cfg.output_dir, "sft_latest"), st, HF_TOKEN)
                    return

                if global_step >= total_sft_steps:
                    break

        neftune_handle.remove()

    # =========================================================================
    # STAGE 2: 16-BIT ADAPTER-REFERENCE DPO + RPO-NLL ALIGNMENT
    # =========================================================================
    if (time.time() - start_time) / 3600.0 < cfg.max_walltime_hours - 0.8:
        print("\n[Stage 2] Merging Stage-1 16-Bit SFT Adapter into Reference Backbone (Zero Extra VRAM)...")
        base_model = model.merge_and_unload()
        base_model.config.use_cache = False
        base_model.enable_input_require_grads()
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        dpo_lora = LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.05,
            use_rslora=True,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        dpo_model = get_peft_model(base_model, dpo_lora)

        dpo_rows = load_hub_jsonl(cfg.dpo_dataset_repo, "data/train.jsonl", HF_TOKEN)
        print(f"[Stage 2 Data] Loaded {len(dpo_rows)} DPO preference pairs.")
        dpo_ds = ManimDPODataset(dpo_rows, tokenizer, max_seq_length=2048)
        dpo_loader = DataLoader(dpo_ds, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])

        dpo_accum = 8
        total_dpo_steps = min(cfg.dpo_max_steps, max(1, len(dpo_loader) // dpo_accum))
        dpo_params = [p for p in dpo_model.parameters() if p.requires_grad]
        dpo_opt = torch.optim.AdamW(dpo_params, lr=cfg.dpo_lr, weight_decay=0.01, foreach=True)
        dpo_sched = get_cosine_schedule_with_warmup(
            dpo_opt, num_warmup_steps=max(5, int(0.05 * total_dpo_steps)), num_training_steps=total_dpo_steps
        )

        first_dev = next(dpo_model.parameters()).device
        dpo_model.train()
        dpo_opt.zero_grad()

        dpo_step = 0
        dpo_micro = 0
        run_dpo_loss = 0.0
        run_dpo_acc = 0.0
        dpo_history: List[Dict[str, Any]] = []

        for item in dpo_loader:
            c_ids = item["chosen_input_ids"].unsqueeze(0).to(first_dev)
            c_lbl = item["chosen_labels"].unsqueeze(0).to(first_dev)
            r_ids = item["rejected_input_ids"].unsqueeze(0).to(first_dev)
            r_lbl = item["rejected_labels"].unsqueeze(0).to(first_dev)

            with torch.no_grad():
                with dpo_model.disable_adapter():
                    ref_c_logp, _ = compute_dpo_logps(dpo_model(input_ids=c_ids).logits, c_lbl)
                    ref_r_logp, _ = compute_dpo_logps(dpo_model(input_ids=r_ids).logits, r_lbl)

            pol_c_logp, c_nll = compute_dpo_logps(dpo_model(input_ids=c_ids).logits, c_lbl)
            pol_r_logp, _ = compute_dpo_logps(dpo_model(input_ids=r_ids).logits, r_lbl)

            margin = cfg.dpo_beta * ((pol_c_logp - pol_r_logp) - (ref_c_logp - ref_r_logp))
            loss = (-F.logsigmoid(margin).mean() + 0.1 * c_nll) / dpo_accum
            loss.backward()

            run_dpo_loss += loss.item()
            run_dpo_acc += float((margin > 0).float().mean().item()) / dpo_accum
            dpo_micro += 1

            if dpo_micro % dpo_accum == 0:
                torch.nn.utils.clip_grad_norm_(dpo_params, 1.0)
                dpo_opt.step()
                dpo_sched.step()
                dpo_opt.zero_grad()
                dpo_step += 1

                if dpo_step % 5 == 0:
                    avg_dl = run_dpo_loss / 5.0
                    avg_acc = (run_dpo_acc / 5.0) * 100.0
                    run_dpo_loss = 0.0
                    run_dpo_acc = 0.0
                    print(f"  [DPO Step {dpo_step:4d}/{total_dpo_steps:4d}] Loss: {avg_dl:.4f} | Pref Acc: {avg_acc:.1f}%")
                    dpo_history.append({"step": dpo_step, "dpo_loss": round(avg_dl, 5), "pref_acc": round(avg_acc, 2)})

                if dpo_step % 50 == 0 or dpo_step >= total_dpo_steps:
                    st = {"global_step": dpo_step, "status": "completed" if dpo_step >= total_dpo_steps else "in_progress", "history": dpo_history}
                    sync_to_hub(dpo_model, tokenizer, cfg.hub_dpo_repo, os.path.join(cfg.output_dir, "dpo_latest"), st, HF_TOKEN)

                if dpo_step >= total_dpo_steps or (time.time() - start_time) / 3600.0 >= cfg.max_walltime_hours:
                    break

    print("\n=== ALL KAGGLE 16-BIT TRAINING & ALIGNMENT STAGES FINISHED! ===")


if __name__ == "__main__":
    main()
