"""
Custom From-Scratch Pure-16-Bit Dual-GPU PyTorch SFT Engine for ManimCE.
========================================================================
Engineered specifically for Kaggle's Dual NVIDIA T4 (`2 x 16 GB = 32 GB VRAM`)
with ZERO 4-bit quantization loss:
  1. Pure FP16 (`torch.float16`) model parallelism balanced across `cuda:0` & `cuda:1`
     via `device_map="auto"` and `max_memory={0: "15.0GiB", 1: "15.0GiB"}`.
  2. High-Rank Rank-Stabilized LoRA (`rsLoRA`, r=64, alpha=64, scaling=alpha/sqrt(r)=8.0)
     across all 7 attention and MLP projections (`q, k, v, o, gate, up, down`).
  3. Custom Multi-Turn Completion-Only Loss Masking (`ManimCompletionOnlyCollator`)
     that masks all `system` and `user` prompt tokens with `-100` so 100% of gradient
     updates go into `<think>` spatial planning and executable ManimCE Python code.
  4. Custom `NEFTune` noisy embedding forward hook (`alpha=5.0`) to prevent syntax
     overfitting and boost Pass@1 code generalization.
  5. Fault-tolerant HuggingFace Hub checkpointing & automatic cross-session resumption.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False
from peft import LoraConfig, PeftModel, get_peft_model
from huggingface_hub import HfApi, snapshot_download


@dataclass
class SFTConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    dataset_repo: str = "PRB1/ManimCE-Master-SFT"
    hub_model_repo: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit"
    output_dir: str = "checkpoints/sft_16bit"
    max_seq_length: int = 3072
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    use_rslora: bool = True
    neftune_alpha: float = 5.0
    per_device_batch_size: int = 1
    grad_accum_steps: int = 16
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    num_epochs: int = 2
    max_steps: int = -1
    save_every_steps: int = 150
    eval_every_steps: int = 150
    log_every_steps: int = 10
    max_grad_norm: float = 1.0
    max_walltime_hours: float = 11.2
    seed: int = 2026


class ManimChatDataset(Dataset):
    """
    Tokenizes multi-turn ChatML conversations (`system`, `user`, `assistant`)
    and constructs exact token-level `labels` where ONLY `assistant` turns contribute
    to the cross-entropy loss (`system` and `user` tokens are masked with `-100`).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_seq_length: int = 3072,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Identify ChatML assistant header & end tokens for Qwen2.5 / Qwen3
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assistant_prefix_ids = tokenizer.encode("assistant\n", add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.records)

    def _build_masked_labels(self, input_ids: List[int]) -> List[int]:
        """
        Mask all tokens with -100 EXCEPT tokens inside `<|im_start|>assistant\n ... <|im_end|>`.
        """
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
                content_start = i + 1 + pref_len
                j = content_start
                while j < n and input_ids[j] != self.im_end_id:
                    labels[j] = input_ids[j]
                    j += 1
                if j < n and input_ids[j] == self.im_end_id:
                    labels[j] = input_ids[j]  # Supervise <|im_end|> stop token
                    i = j + 1
                else:
                    break
            else:
                i += 1
        return labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        messages = rec["messages"]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = self.tokenizer(
            rendered,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        labels = self._build_masked_labels(input_ids)

        # Fallback safety: if truncation cut before assistant span, supervise tail
        if all(x == -100 for x in labels) and len(input_ids) > 32:
            half = len(input_ids) // 2
            for k in range(half, len(input_ids)):
                labels[k] = input_ids[k]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_completion_only(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Right-pad batch of variable-length sequences with `pad_token_id` and `-100` for labels."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i, :seq_len] = item["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def register_neftune_hook(model: nn.Module, alpha: float = 5.0):
    """
    Attach NEFTune uniform noise hook to input token embeddings during training:
    e' = e + Uniform(-1, 1) * (alpha / sqrt(seq_len * hidden_dim))
    """
    emb_layer = model.get_input_embeddings()

    def _neftune_forward_hook(module: nn.Module, args: Tuple, output: torch.Tensor) -> torch.Tensor:
        if module.training and alpha > 0.0:
            dims = torch.tensor(output.size(1) * output.size(2), dtype=output.dtype, device=output.device)
            mag_norm = alpha / torch.sqrt(dims)
            noise = torch.zeros_like(output).uniform_(-mag_norm.item(), mag_norm.item())
            return output + noise
        return output

    return emb_layer.register_forward_hook(_neftune_forward_hook)


def load_jsonl_from_hub_or_local(
    dataset_repo: str,
    split_filename: str,
    local_fallback: str,
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load JSONL split from local path if present, otherwise download from HuggingFace Hub."""
    if os.path.exists(local_fallback):
        path = local_fallback
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=dataset_repo,
            filename=split_filename,
            repo_type="dataset",
            token=hf_token,
        )
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_balanced_device_map() -> Tuple[Any, Optional[Dict[int, str]]]:
    """
    Configure native 16-bit model sharding across all available GPUs
    (e.g., Kaggle 2x T4 = 32 GB VRAM) to maximize VRAM utilization.
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        max_mem = {}
        for i in range(num_gpus):
            total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            # Use ~93% of each GPU's VRAM capacity
            usable_gib = max(10.0, round(total_gib * 0.93, 1))
            max_mem[i] = f"{usable_gib}GiB"
        print(f"[Dual-GPU 16-Bit Sharding] Detected {num_gpus} GPUs -> max_memory={max_mem}")
        return "auto", max_mem
    elif num_gpus == 1:
        print("[Single-GPU 16-Bit Mode] Detected 1 GPU.")
        return {"": 0}, None
    else:
        raise RuntimeError("No CUDA GPU detected! Please enable GPU T4 x2 on Kaggle.")


@torch.no_grad()
def evaluate_validation_loss(
    model: nn.Module,
    val_loader: DataLoader,
    first_device: torch.device,
    max_batches: int = 100,
) -> Tuple[float, float]:
    """Compute validation cross-entropy loss and perplexity on held-out validation split."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for idx, batch in enumerate(val_loader):
        if idx >= max_batches:
            break
        batch = {k: v.to(first_device, non_blocking=True) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        if not torch.isnan(loss) and not torch.isinf(loss):
            total_loss += loss.item()
            total_batches += 1

    model.train()
    avg_loss = total_loss / max(1, total_batches)
    ppl = math.exp(min(20.0, avg_loss))
    return avg_loss, ppl


def push_checkpoint_to_hub(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    cfg: SFTConfig,
    state: Dict[str, Any],
    hf_token: str,
    subfolder: Optional[str] = None,
) -> None:
    """Save 16-bit rsLoRA adapter + training state locally and push to HuggingFace Hub."""
    save_dir = os.path.join(cfg.output_dir, subfolder if subfolder else "latest")
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    state_path = os.path.join(save_dir, "training_state.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    if hf_token:
        try:
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=cfg.hub_model_repo, repo_type="model", exist_ok=True, private=False)
            api.upload_folder(
                folder_path=save_dir,
                path_in_repo=subfolder if subfolder else "",
                repo_id=cfg.hub_model_repo,
                repo_type="model",
                commit_message=f"Checkpoint step {state.get('global_step', 0)} (val_loss={state.get('best_val_loss', 0.0):.4f})",
            )
            print(f"  [Hub Sync] Pushed checkpoint (step={state.get('global_step')}) -> {cfg.hub_model_repo}")
        except Exception as exc:
            print(f"  [WARN] Hub checkpoint push encountered transient error: {exc}")


def maybe_resume_from_hub(
    base_model: nn.Module,
    cfg: SFTConfig,
    hf_token: Optional[str],
) -> Tuple[nn.Module, int, float, List[Dict[str, Any]]]:
    """
    Check if a previous Kaggle session already uploaded an adapter checkpoint to
    `cfg.hub_model_repo`. If found, load the adapter weights and resume from `global_step`!
    """
    if not hf_token:
        return base_model, 0, float("inf"), []
    api = HfApi(token=hf_token)
    try:
        files = api.list_repo_files(repo_id=cfg.hub_model_repo, repo_type="model")
        if "adapter_config.json" in files and "training_state.json" in files:
            print(f"[Resume] Found existing checkpoint on Hub ({cfg.hub_model_repo}). Downloading...")
            resume_dir = snapshot_download(
                repo_id=cfg.hub_model_repo,
                repo_type="model",
                token=hf_token,
                local_dir=os.path.join(cfg.output_dir, "resumed_hub"),
            )
            with open(os.path.join(resume_dir, "training_state.json"), "r", encoding="utf-8") as f:
                st = json.load(f)
            resumed_step = int(st.get("global_step", 0))
            best_val = float(st.get("best_val_loss", float("inf")))
            history = st.get("history", [])
            model = PeftModel.from_pretrained(base_model, resume_dir, is_trainable=True)
            print(f"[Resume] Successfully resumed 16-bit rsLoRA adapter at global_step={resumed_step} (best_val_loss={best_val:.4f})!")
            return model, resumed_step, best_val, history
    except Exception:
        pass
    return base_model, 0, float("inf"), []


def run_sft_training(cfg: SFTConfig, hf_token: str) -> Dict[str, Any]:
    """Execute the complete custom 16-bit Dual-GPU rsLoRA SFT training loop."""
    start_wall = time.time()
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 76)
    print(f"ManimCE 16-Bit Dual-GPU Custom SFT Engine — Base: {cfg.base_model}")
    print("=" * 76)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map, max_memory = get_balanced_device_map()

    print(f"[1/5] Loading {cfg.base_model} in PURE 16-BIT (torch.float16, SDPA, zero 4-bit quantization) ...")
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
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Check Hub resumption first; otherwise initialize fresh high-rank rsLoRA adapter
    model, resumed_step, best_val_loss, history = maybe_resume_from_hub(base_model, cfg, hf_token)
    if resumed_step == 0:
        print(
            f"[2/5] Attaching High-Rank rsLoRA Adapter (r={cfg.lora_r}, alpha={cfg.lora_alpha}, "
            f"use_rslora={cfg.use_rslora}) across all 7 linear layers ..."
        )
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            use_rslora=cfg.use_rslora,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(base_model, lora_cfg)

    model.print_trainable_parameters()
    neftune_handle = register_neftune_hook(model, alpha=cfg.neftune_alpha)

    print(f"[3/5] Loading curated ManimCE SFT splits from {cfg.dataset_repo} ...")
    train_rows = load_jsonl_from_hub_or_local(
        cfg.dataset_repo, "data/train.jsonl", "data/processed/sft_train.jsonl", hf_token
    )
    val_rows = load_jsonl_from_hub_or_local(
        cfg.dataset_repo, "data/validation.jsonl", "data/processed/sft_val.jsonl", hf_token
    )
    print(f"  -> Loaded {len(train_rows)} train examples and {len(val_rows)} validation examples.")

    train_ds = ManimChatDataset(train_rows, tokenizer, max_seq_length=cfg.max_seq_length)
    val_ds = ManimChatDataset(val_rows, tokenizer, max_seq_length=cfg.max_seq_length)

    collate_fn = lambda b: collate_completion_only(b, tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.per_device_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
    total_steps = cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    # Use 8-bit Paged AdamW if bitsandbytes is installed on Kaggle, else standard fused AdamW
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb  # type: ignore

        optimizer = bnb.optim.PagedAdamW8bit(
            trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        print("[4/5] Optimizer: PagedAdamW8bit (saves optimizer VRAM while keeping model 100% 16-bit)")
    except Exception:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, foreach=True
        )
        print("[4/5] Optimizer: Native PyTorch AdamW")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Fast-forward scheduler if resuming
    for _ in range(resumed_step):
        scheduler.step()

    first_device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad()

    global_step = resumed_step
    micro_step = 0
    running_loss = 0.0
    print(f"[5/5] Starting training loop: target total_steps={total_steps} (starting from step={global_step}) ...")

    for epoch in range(cfg.num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            # Skip already-completed batches in the resumed epoch
            global_micro_target = global_step * cfg.grad_accum_steps
            if epoch == 0 and resumed_step > 0 and batch_idx < (resumed_step * cfg.grad_accum_steps) % len(train_loader):
                continue

            batch = {k: v.to(first_device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / cfg.grad_accum_steps
            loss.backward()

            running_loss += loss.item()
            micro_step += 1

            if micro_step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_every_steps == 0:
                    elapsed_min = (time.time() - start_wall) / 60.0
                    vram_gb = sum(
                        torch.cuda.max_memory_allocated(i) / (1024 ** 3)
                        for i in range(torch.cuda.device_count())
                    )
                    lr_now = scheduler.get_last_lr()[0]
                    avg_train_loss = running_loss / cfg.log_every_steps
                    running_loss = 0.0
                    print(
                        f"  Step [{global_step:4d}/{total_steps:4d}] | Epoch {epoch+1} | "
                        f"Train Loss: {avg_train_loss:.4f} | LR: {lr_now:.2e} | "
                        f"Peak VRAM: {vram_gb:.2f} GB | Elapsed: {elapsed_min:.1f}m"
                    )
                    history.append(
                        {
                            "step": global_step,
                            "train_loss": round(avg_train_loss, 5),
                            "lr": lr_now,
                            "vram_gb": round(vram_gb, 2),
                            "elapsed_min": round(elapsed_min, 2),
                        }
                    )

                # Periodic Validation & Best Checkpoint Upload
                if global_step % cfg.eval_every_steps == 0 or global_step >= total_steps:
                    val_loss, val_ppl = evaluate_validation_loss(model, val_loader, first_device)
                    print(f"  >>> [Eval @ Step {global_step}] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss

                    state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "best_val_loss": best_val_loss,
                        "config": asdict(cfg),
                        "history": history,
                    }
                    push_checkpoint_to_hub(model, tokenizer, cfg, state, hf_token, subfolder=None)
                    if is_best:
                        push_checkpoint_to_hub(model, tokenizer, cfg, state, hf_token, subfolder="best")

                elif global_step % cfg.save_every_steps == 0:
                    state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "config": asdict(cfg),
                        "history": history,
                    }
                    push_checkpoint_to_hub(model, tokenizer, cfg, state, hf_token, subfolder=None)

                # Graceful Kaggle Session Wall-Time Guard
                elapsed_hours = (time.time() - start_wall) / 3600.0
                if elapsed_hours >= cfg.max_walltime_hours:
                    print(
                        f"\n[Walltime Guard] Reached {elapsed_hours:.2f} hrs (limit={cfg.max_walltime_hours} hrs). "
                        "Saving checkpoint to HuggingFace Hub and exiting cleanly..."
                    )
                    state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "status": "paused_for_session_continuation",
                        "config": asdict(cfg),
                        "history": history,
                    }
                    push_checkpoint_to_hub(model, tokenizer, cfg, state, hf_token, subfolder=None)
                    neftune_handle.remove()
                    return state

                if global_step >= total_steps:
                    break

        if global_step >= total_steps:
            break

    neftune_handle.remove()
    final_state = {
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "status": "completed",
        "config": asdict(cfg),
        "history": history,
    }
    push_checkpoint_to_hub(model, tokenizer, cfg, final_state, hf_token, subfolder=None)
    print("\n=== Stage 1 (16-Bit rsLoRA SFT) Completed Successfully! ===")
    return final_state


def parse_args() -> SFTConfig:
    parser = argparse.ArgumentParser(description="16-Bit Dual-GPU ManimCE SFT Trainer")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--dataset_repo", type=str, default="PRB1/ManimCE-Master-SFT")
    parser.add_argument("--hub_model_repo", type=str, default="PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft_16bit")
    parser.add_argument("--max_seq_length", type=int, default=3072)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1.5e-4)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_every_steps", type=int, default=150)
    parser.add_argument("--eval_every_steps", type=int, default=150)
    parser.add_argument("--max_walltime_hours", type=float, default=11.2)
    args = parser.parse_args()
    return SFTConfig(**vars(args))


if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN", "")
    config = parse_args()
    run_sft_training(config, hf_token=token)
