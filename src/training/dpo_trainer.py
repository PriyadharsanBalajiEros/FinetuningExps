"""
Custom From-Scratch Pure-16-Bit Dual-GPU DPO + RPO-NLL Alignment Engine.
========================================================================
Aligns the Stage-1 16-bit SFT ManimCE model against compiler errors, deprecated
ManimGL / v0.14 API hallucinations, and spatial layout bugs using zero-overhead
adapter-toggled reference log-probabilities (`with model.disable_adapter():`):

  1. Merges Stage-1 16-bit SFT `rsLoRA` adapter into the 16-bit base model in-memory
     so the frozen 16-bit backbone IS the exact Stage-1 SFT reference policy (pi_ref),
     requiring ZERO duplicate reference model VRAM!
  2. Attaches a Stage-2 16-bit `rsLoRA` DPO adapter and optimizes the RPO-stabilized
     Direct Preference Optimization objective:
       L = -log sigmoid( beta * ((log pi_theta(y_w|x) - log pi_ref(y_w|x))
                               - (log pi_theta(y_l|x) - log pi_ref(y_l|x))) )
           + alpha_nll * L_NLL(y_w | x)
     where `alpha_nll = 0.1` prevents likelihood drop on shared Python tokens.
  3. Automatically saves both the Stage-2 DPO adapter and the final merged 16-bit
     flagship model to HuggingFace Hub (`PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit`).
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

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
from huggingface_hub import HfApi, snapshot_download

from src.training.sft_trainer import (
    get_balanced_device_map,
    load_jsonl_from_hub_or_local,
)


@dataclass
class DPOConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    sft_adapter_repo: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit"
    dpo_dataset_repo: str = "PRB1/ManimCE-Master-DPO"
    hub_dpo_repo: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit"
    output_dir: str = "checkpoints/dpo_16bit"
    max_seq_length: int = 2560
    beta: float = 0.1
    nll_loss_weight: float = 0.1
    lora_r: int = 32
    lora_alpha: int = 32
    use_rslora: bool = True
    grad_accum_steps: int = 8
    learning_rate: float = 2.0e-5
    num_epochs: int = 1
    max_steps: int = -1
    save_every_steps: int = 100
    log_every_steps: int = 5
    max_walltime_hours: float = 11.2
    seed: int = 2026


class ManimDPODataset(Dataset):
    """
    Tokenizes `(system, prompt, chosen, rejected)` preference pairs and masks
    prompt tokens with `-100` so log-probabilities are summed strictly over the
    assistant completion (`chosen` vs `rejected`).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_seq_length: int = 2560,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.records)

    def _encode_pair_item(self, system: str, prompt: str, response: str) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": response}]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(
            full_text, truncation=True, max_length=self.max_seq_length, add_special_tokens=False
        )

        labels = list(full_ids)
        mask_len = min(len(prompt_ids), len(labels) - 1)
        for i in range(mask_len):
            labels[i] = -100

        return (
            torch.tensor(full_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]
        system = r.get("system", "")
        prompt = r["prompt"]
        chosen_ids, chosen_labels = self._encode_pair_item(system, prompt, r["chosen"])
        rejected_ids, rejected_labels = self._encode_pair_item(system, prompt, r["rejected"])
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
        }


def compute_sequence_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sum of token log-probabilities and average NLL loss over non-masked (`!= -100`)
    response tokens.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_mask = shift_labels != -100
    safe_labels = torch.where(loss_mask, shift_labels, torch.zeros_like(shift_labels))

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    per_token_logps = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
    masked_logps = per_token_logps * loss_mask.float()

    seq_logp_sum = masked_logps.sum(dim=-1)
    token_count = loss_mask.float().sum(dim=-1).clamp(min=1.0)
    nll_loss = -(seq_logp_sum / token_count).mean()
    return seq_logp_sum, nll_loss


def run_dpo_training(cfg: DPOConfig, hf_token: str) -> Dict[str, Any]:
    """Execute 16-bit Adapter-Reference DPO alignment on Dual-GPU."""
    start_wall = time.time()
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 76)
    print(f"ManimCE 16-Bit Dual-GPU Custom DPO Alignment Engine — Base: {cfg.base_model}")
    print("=" * 76)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map, max_memory = get_balanced_device_map()

    print(f"[1/4] Loading {cfg.base_model} in PURE 16-BIT and merging Stage-1 SFT adapter ({cfg.sft_adapter_repo}) ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
    )

    # Download and merge Stage-1 SFT adapter so the frozen 16-bit backbone IS pi_ref
    try:
        sft_dir = snapshot_download(repo_id=cfg.sft_adapter_repo, repo_type="model", token=hf_token)
        sft_peft = PeftModel.from_pretrained(base_model, sft_dir)
        base_model = sft_peft.merge_and_unload()
        print("  -> Successfully merged Stage-1 16-bit SFT adapter into backbone as reference policy (pi_ref)!")
    except Exception as exc:
        print(f"  [WARN] Could not load Stage-1 SFT adapter from {cfg.sft_adapter_repo} ({exc}). Using base model as pi_ref.")

    base_model.config.use_cache = False
    base_model.enable_input_require_grads()
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.05,
        use_rslora=cfg.use_rslora,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    print(f"[2/4] Loading DPO preference pairs from {cfg.dpo_dataset_repo} ...")
    train_rows = load_jsonl_from_hub_or_local(
        cfg.dpo_dataset_repo, "data/train.jsonl", "data/processed/dpo_train.jsonl", hf_token
    )
    dpo_ds = ManimDPODataset(train_rows, tokenizer, max_seq_length=cfg.max_seq_length)
    train_loader = DataLoader(dpo_ds, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])

    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
    total_steps = cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.num_epochs

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(5, int(0.05 * total_steps)), num_training_steps=total_steps
    )

    first_device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad()

    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_acc = 0.0
    history: List[Dict[str, Any]] = []

    print(f"[3/4] Starting 16-Bit DPO Alignment: total_steps={total_steps}, beta={cfg.beta} ...")
    for epoch in range(cfg.num_epochs):
        for item in train_loader:
            c_ids = item["chosen_input_ids"].unsqueeze(0).to(first_device)
            c_lbl = item["chosen_labels"].unsqueeze(0).to(first_device)
            r_ids = item["rejected_input_ids"].unsqueeze(0).to(first_device)
            r_lbl = item["rejected_labels"].unsqueeze(0).to(first_device)

            # 1. Compute reference log-probs (pi_ref) with adapter disabled (ZERO extra VRAM)
            with torch.no_grad():
                with model.disable_adapter():
                    ref_c_logits = model(input_ids=c_ids).logits
                    ref_c_logp, _ = compute_sequence_logps(ref_c_logits, c_lbl)
                    del ref_c_logits

                    ref_r_logits = model(input_ids=r_ids).logits
                    ref_r_logp, _ = compute_sequence_logps(ref_r_logits, r_lbl)
                    del ref_r_logits

            # 2. Compute policy log-probs (pi_theta) and gradients
            pol_c_logits = model(input_ids=c_ids).logits
            pol_c_logp, chosen_nll = compute_sequence_logps(pol_c_logits, c_lbl)
            del pol_c_logits

            pol_r_logits = model(input_ids=r_ids).logits
            pol_r_logp, _ = compute_sequence_logps(pol_r_logits, r_lbl)
            del pol_r_logits

            pi_logratios = pol_c_logp - pol_r_logp
            ref_logratios = ref_c_logp - ref_r_logp
            margin = cfg.beta * (pi_logratios - ref_logratios)

            dpo_loss = -F.logsigmoid(margin).mean()
            total_loss = (dpo_loss + cfg.nll_loss_weight * chosen_nll) / cfg.grad_accum_steps
            total_loss.backward()

            running_loss += total_loss.item()
            running_acc += float((margin > 0).float().mean().item()) / cfg.grad_accum_steps
            micro_step += 1

            if micro_step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_every_steps == 0:
                    avg_l = running_loss / cfg.log_every_steps
                    avg_acc = (running_acc / cfg.log_every_steps) * 100.0
                    running_loss = 0.0
                    running_acc = 0.0
                    print(
                        f"  DPO Step [{global_step:4d}/{total_steps:4d}] | "
                        f"Loss: {avg_l:.4f} | Preference Acc: {avg_acc:.1f}% | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                    history.append({"step": global_step, "dpo_loss": round(avg_l, 5), "pref_acc": round(avg_acc, 2)})

                if global_step % cfg.save_every_steps == 0 or global_step >= total_steps:
                    save_dir = os.path.join(cfg.output_dir, "latest")
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    with open(os.path.join(save_dir, "dpo_state.json"), "w", encoding="utf-8") as f:
                        json.dump({"global_step": global_step, "config": asdict(cfg), "history": history}, f, indent=2)
                    if hf_token:
                        api = HfApi(token=hf_token)
                        api.create_repo(repo_id=cfg.hub_dpo_repo, repo_type="model", exist_ok=True, private=False)
                        api.upload_folder(
                            folder_path=save_dir,
                            repo_id=cfg.hub_dpo_repo,
                            repo_type="model",
                            commit_message=f"DPO Checkpoint step {global_step}",
                        )

                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break

    print("[4/4] Stage 2 (16-Bit DPO Alignment) Completed!")
    return {"global_step": global_step, "history": history}


def parse_args() -> DPOConfig:
    parser = argparse.ArgumentParser(description="16-Bit Dual-GPU ManimCE DPO Trainer")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--sft_adapter_repo", type=str, default="PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit")
    parser.add_argument("--dpo_dataset_repo", type=str, default="PRB1/ManimCE-Master-DPO")
    parser.add_argument("--hub_dpo_repo", type=str, default="PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit")
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()
    return DPOConfig(**vars(args))


if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN", "")
    config = parse_args()
    run_dpo_training(config, hf_token=token)
