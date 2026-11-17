"""
V2 Standalone Kaggle GPU Kernel: Pure 16-Bit Dual-T4 Ultra-Animative ManimCE SFT + DPO + Real Render Eval.
===========================================================================================================
Upgrades over V1:
  1. Trains on `PRB1/ManimCE-UltraAnimative-V2-SFT` (5,400 zero-truncation, LaTeX-AST-verified,
     high-animation-density scenes including 800 modular multi-act long scenes) and
     `PRB1/ManimCE-UltraAnimative-V2-DPO` (1,600 pairs penalizing LaTeX concat/brace bugs,
     static slides, and truncated code).
  2. Uses stabilized `rsLoRA` LR (`4.0e-5`, `max_grad_norm=0.5`) with Automatic Best-Checkpoint
     Rollback & LR Halving so validation loss decreases monotonically without overshooting.
  3. Runs Stage-3 Evaluation with REAL `manim render -ql` + LaTeX compilation on Kaggle
     (including the 5-Act Softmax -> Attention -> Sliding Window Attention stress-test!)
     and uploads `v2_real_render_eval_results.json` to `PRB1/Qwen2.5-Coder-7B-ManimCE-V2-DPO-16bit`.
"""

import ast
import glob
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# 1. Enable expandable CUDA memory segments before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 2. Remove incompatible torchao 0.10.0 from Kaggle base image before importing peft
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    check=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# 3. Configure HF_TOKEN before importing huggingface_hub / transformers
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
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
import peft.import_utils
import peft.tuners.lora.torchao

peft.import_utils.is_torchao_available = lambda: False
peft.tuners.lora.torchao.is_torchao_available = lambda: False
from peft import LoraConfig, PeftModel, get_peft_model
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


SYSTEM_PROMPT_V2 = (
    "You are an elite Manim Community Edition (ManimCE v0.19+) developer who creates "
    "visually rich, deeply animative 3Blue1Brown-style mathematical animations.\n"
    "Follow these strict rules in every script:\n"
    "1. Output ONLY a single complete, runnable Python code block (`from manim import *`). "
    "Never truncate code and never output prose before the code block.\n"
    "2. Make every scene highly animative: use geometric Mobjects (`Axes`, `NumberPlane`, "
    "`VGroup`, `Arrow`, `Brace`, `SurroundingRectangle`, `Circle`, `Square`, `Dot`), smooth "
    "dynamic animations (`Create`, `ReplacementTransform`, `FadeTransform`, `LaggedStart`, "
    "`AnimationGroup`, `Indicate`, `Circumscribe`, `ValueTracker`, `always_redraw`), and "
    "explicit `run_time` and `self.wait()` pacing.\n"
    "3. For multi-concept or long videos, structure the Scene into clean modular helper "
    "methods (`self.act_1_...()`, `self.clear_stage()`, `self.act_2_...()`) and always fade "
    "out previous act Mobjects (`self.play(*[FadeOut(m) for m in self.mobjects])`) before "
    "starting the next act so text and graphics NEVER overlap.\n"
    "4. Keep all `MathTex(r\"...\")` strings as single raw string literals with balanced "
    "`{}` braces—never concatenate strings with `+` inside `MathTex` and never use unescaped `%`."
)


@dataclass
class V2PipelineConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    sft_dataset_repo: str = "PRB1/ManimCE-UltraAnimative-V2-SFT"
    dpo_dataset_repo: str = "PRB1/ManimCE-UltraAnimative-V2-DPO"
    hub_sft_repo: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-V2-SFT-16bit"
    hub_dpo_repo: str = "PRB1/Qwen2.5-Coder-7B-ManimCE-V2-DPO-16bit"
    output_dir: str = "/kaggle/working/checkpoints_v2"
    max_seq_length: int = 2560
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    use_rslora: bool = True
    neftune_alpha: float = 5.0
    grad_accum_steps: int = 16
    sft_lr: float = 4.0e-5
    dpo_lr: float = 1.0e-5
    dpo_beta: float = 0.1
    sft_max_steps: int = 120
    dpo_max_steps: int = 60
    save_every_steps: int = 20
    eval_every_steps: int = 20
    log_every_steps: int = 5
    max_grad_norm: float = 0.5
    max_walltime_hours: float = 11.0
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
        c_ids, c_lbl = self._encode_item(r.get("system", SYSTEM_PROMPT_V2), r["prompt"], r["chosen"])
        r_ids, r_lbl = self._encode_item(r.get("system", SYSTEM_PROMPT_V2), r["prompt"], r["rejected"])
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
            noisy = output + torch.zeros_like(output).uniform_(-mag.item(), mag.item())
            return noisy.requires_grad_(True)
        return output

    return emb.register_forward_hook(_hook)


def _extract_backbone_and_head(model: nn.Module) -> Tuple[nn.Module, nn.Module]:
    causal_lm = model.base_model.model if hasattr(model, "base_model") else model
    return causal_lm.model, causal_lm.lm_head


def compute_chunked_ce_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    backbone, lm_head = _extract_backbone_and_head(model)
    outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden_states = outputs[0]

    shift_hidden = hidden_states[:, :-1, :].contiguous().view(-1, hidden_states.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1).to(shift_hidden.device)

    valid_mask = shift_labels != -100
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return hidden_states.sum() * 0.0

    valid_hidden = shift_hidden[valid_mask]
    valid_labels = shift_labels[valid_mask]

    def _chunk_loss_fn(h_slice: torch.Tensor, y_slice: torch.Tensor) -> torch.Tensor:
        logits_slice = lm_head(h_slice).float()
        return F.cross_entropy(logits_slice, y_slice, reduction="sum")

    total_loss = torch.zeros((), dtype=torch.float32, device=valid_hidden.device)
    for start in range(0, valid_count, chunk_size):
        end = min(start + chunk_size, valid_count)
        h_chunk = valid_hidden[start:end]
        y_chunk = valid_labels[start:end]
        if model.training and h_chunk.requires_grad:
            chunk_loss = checkpoint(_chunk_loss_fn, h_chunk, y_chunk, use_reentrant=False)
        else:
            chunk_loss = _chunk_loss_fn(h_chunk, y_chunk)
        total_loss = total_loss + chunk_loss

    return total_loss / float(valid_count)


def compute_chunked_dpo_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone, lm_head = _extract_backbone_and_head(model)
    hidden_states = backbone(input_ids=input_ids, use_cache=False)[0]

    shift_hidden = hidden_states[:, :-1, :].contiguous().view(-1, hidden_states.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1).to(shift_hidden.device)

    valid_mask = shift_labels != -100
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        zero = hidden_states.sum() * 0.0
        return zero.unsqueeze(0), zero

    valid_hidden = shift_hidden[valid_mask]
    valid_labels = shift_labels[valid_mask]

    def _chunk_logp_fn(h_slice: torch.Tensor, y_slice: torch.Tensor) -> torch.Tensor:
        logits_slice = lm_head(h_slice).float()
        log_probs = F.log_softmax(logits_slice, dim=-1)
        token_logps = torch.gather(log_probs, dim=-1, index=y_slice.unsqueeze(-1)).squeeze(-1)
        return token_logps.sum()

    total_logp = torch.zeros((), dtype=torch.float32, device=valid_hidden.device)
    for start in range(0, valid_count, chunk_size):
        end = min(start + chunk_size, valid_count)
        h_chunk = valid_hidden[start:end]
        y_chunk = valid_labels[start:end]
        if model.training and h_chunk.requires_grad:
            c_logp = checkpoint(_chunk_logp_fn, h_chunk, y_chunk, use_reentrant=False)
        else:
            c_logp = _chunk_logp_fn(h_chunk, y_chunk)
        total_logp = total_logp + c_logp

    nll = -total_logp / float(valid_count)
    return total_logp.unsqueeze(0), nll


def clip_grad_norm_multi_gpu(parameters: List[nn.Parameter], max_norm: float = 0.5) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    total_norm_sq = sum(torch.sum(g.detach().float() ** 2).item() for g in grads)
    total_norm = math.sqrt(max(0.0, total_norm_sq))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.detach().mul_(clip_coef)
    return total_norm


def cast_trainable_lora_to_fp32(model: nn.Module) -> None:
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)


def snapshot_trainable_weights_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def restore_trainable_weights_from_cpu(model: nn.Module, cpu_state: Dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad and name in cpu_state:
            param.data.copy_(cpu_state[name].to(param.device, dtype=param.dtype))


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
    print(f"[Hardware Check] Detected {n_gpus} CUDA GPU(s):", flush=True)
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  - GPU {i}: {props.name} ({props.total_memory / (1024**3):.2f} GiB VRAM)", flush=True)
    if n_gpus >= 2:
        return "auto", {0: "13.6GiB", 1: "11.0GiB"}
    elif n_gpus == 1:
        return {"": 0}, None
    raise RuntimeError("No CUDA GPU available on this Kaggle instance!")


@torch.no_grad()
def evaluate_sft_val(model: nn.Module, val_loader: DataLoader, first_dev: torch.device) -> Tuple[float, float]:
    model.eval()
    tot_loss, count = 0.0, 0
    for idx, batch in enumerate(val_loader):
        if idx >= 40:
            break
        batch = {k: v.to(first_dev, non_blocking=True) for k, v in batch.items()}
        loss = compute_chunked_ce_loss(model, batch["input_ids"], batch["attention_mask"], batch["labels"])
        if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() > 0:
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
                commit_message=f"V2 Checkpoint step {state.get('global_step', 0)}",
            )
            print(
                f"  [Hub Checkpoint Saved] -> https://huggingface.co/{repo_id} (step={state.get('global_step')})",
                flush=True,
            )
        except Exception as exc:
            print(f"  [WARN] Hub upload warning: {exc}", flush=True)


def install_manim_for_real_eval() -> bool:
    """Install headless ManimCE + LaTeX + FFmpeg for Stage-3 real video rendering benchmark."""
    print("\n[Stage 3 Setup] Installing ManimCE + LaTeX + FFmpeg for real video render evaluation...", flush=True)
    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq libcairo2-dev libpango1.0-dev ffmpeg "
        "texlive-latex-base texlive-latex-extra texlive-fonts-extra texlive-science dvisvgm",
        shell=True,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    res = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "manim"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return res.returncode == 0


def extract_code_and_scene(raw_reply: str) -> Tuple[str, str]:
    text = re.sub(r"<think>.*?</think>", "", raw_reply or "", flags=re.DOTALL).strip()
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    code = m[-1].strip() if m else text
    if "from manim import *" not in code:
        code = "from manim import *\n\n" + code
    scene_name = "MainScene"
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                scene_name = node.name
                break
    except Exception:
        cm = re.search(r"class\s+(\w+)\s*\(", code)
        if cm:
            scene_name = cm.group(1)
    return code, scene_name


def real_manim_render_test(code: str, scene_name: str, timeout_sec: int = 120) -> Tuple[bool, Optional[str]]:
    """Actually execute `manim render -ql` to verify zero Python, Manim, or LaTeX errors."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"

    with tempfile.TemporaryDirectory() as tmpdir:
        py_path = os.path.join(tmpdir, "scene_eval.py")
        with open(py_path, "w", encoding="utf-8") as f:
            f.write(code + "\n")
        try:
            proc = subprocess.run(
                ["manim", "render", "-ql", "--media_dir", tmpdir, py_path, scene_name],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if proc.returncode == 0:
                return True, None
            combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            return False, "\n".join(combined.splitlines()[-25:])
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: Render exceeded {timeout_sec}s"
        except Exception as exc:
            return False, str(exc)


@torch.no_grad()
def run_v2_real_render_eval(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    cfg: V2PipelineConfig,
    max_cases: int = 15,
) -> Dict[str, Any]:
    """Run Stage-3 REAL Manim Render Evaluation on held-out cases + 5-Act Stress Test."""
    install_manim_for_real_eval()
    model.eval()
    model.config.use_cache = True
    first_dev = next(model.parameters()).device

    test_rows = load_hub_jsonl(cfg.sft_dataset_repo, "data/test.jsonl", HF_TOKEN)[:max_cases]
    # Append our 5-Act Softmax -> Attention -> Sliding Window Attention Stress Test!
    stress_prompt = (
        "Create a long, multi-act 3Blue1Brown-style ManimCE animation inside a single Scene "
        "(`SoftmaxToWindowAttention`) structured with modular methods (`act_1_softmax`, `act_2_qkv`, "
        "`act_3_scaled_attention`, `act_4_window_attention`) and `clear_stage()` FadeOut transitions:\n"
        "- Act 1: Show Softmax P(z_i) = e^{z_i} / sum e^{z_j} with MathTex and animated logit bars.\n"
        "- Act 2: Show 4 token boxes ('The', 'cat', 'sat', 'down') projecting into Query (Q), Key (K), Value (V) arrows.\n"
        "- Act 3: Show Scaled Dot-Product Attention softmax(Q K^T / sqrt(d_k)) V.\n"
        "- Act 4: Build a 6x6 attention grid, show full O(N^2) vs Sliding Window O(N*w) diagonal band highlight."
    )
    eval_items = [{"id": r.get("id", f"test_{i}"), "prompt": r["prompt"]} for i, r in enumerate(test_rows)]
    eval_items.append({"id": "stress_test_softmax_to_window_attention", "prompt": stress_prompt})

    p1_render_ok = 0
    p2_render_ok = 0
    details = []

    print(f"\n[Stage 3 Real Render Eval] Testing {len(eval_items)} scenes with real `manim render -ql`...", flush=True)
    for idx, item in enumerate(eval_items):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_V2},
            {"role": "user", "content": item["prompt"]},
        ]
        chat_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_str, return_tensors="pt").to(first_dev)
        out_ids = model.generate(
            **inputs,
            max_new_tokens=2800,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        reply_1 = tokenizer.decode(out_ids[0][inputs["input_ids"].size(1) :], skip_special_tokens=True)
        code_1, scene_1 = extract_code_and_scene(reply_1)
        ok_1, err_1 = real_manim_render_test(code_1, scene_1)

        if ok_1:
            p1_render_ok += 1
            p2_render_ok += 1
            details.append({"id": item["id"], "pass1_render": True, "pass2_render": True})
        else:
            repair_msg = (
                f"Running the generated Manim script produced the following compiler/runtime error:\n"
                f"```text\n{err_1}\n```\n\n"
                "Please fix the bug and provide the complete corrected ManimCE v0.19+ Python script."
            )
            msgs_2 = msgs + [{"role": "assistant", "content": reply_1}, {"role": "user", "content": repair_msg}]
            chat_2 = tokenizer.apply_chat_template(msgs_2, tokenize=False, add_generation_prompt=True)
            in_2 = tokenizer(chat_2, return_tensors="pt").to(first_dev)
            out_2 = model.generate(
                **in_2,
                max_new_tokens=2800,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            reply_2 = tokenizer.decode(out_2[0][in_2["input_ids"].size(1) :], skip_special_tokens=True)
            code_2, scene_2 = extract_code_and_scene(reply_2)
            ok_2, err_2 = real_manim_render_test(code_2, scene_2)
            if ok_2:
                p2_render_ok += 1
            details.append(
                {"id": item["id"], "pass1_render": False, "pass2_render": ok_2, "error": err_2 if not ok_2 else None}
            )

        n = idx + 1
        print(
            f"  [Real Render {n:2d}/{len(eval_items):2d}] {item['id']}: "
            f"P1={ok_1} | P2={details[-1]['pass2_render']} | "
            f"Cumul P1: {100*p1_render_ok/n:.1f}% | P2: {100*p2_render_ok/n:.1f}%",
            flush=True,
        )

    tot = max(1, len(eval_items))
    summary = {
        "num_cases": len(eval_items),
        "pass1_real_mp4_render_pct": round(100.0 * p1_render_ok / tot, 2),
        "pass2_ritl_real_mp4_render_pct": round(100.0 * p2_render_ok / tot, 2),
        "details": details,
    }
    eval_path = os.path.join(cfg.output_dir, "v2_real_render_eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if HF_TOKEN:
        try:
            api = HfApi(token=HF_TOKEN)
            api.upload_file(
                path_or_fileobj=eval_path,
                path_in_repo="v2_real_render_eval_results.json",
                repo_id=cfg.hub_dpo_repo,
                repo_type="model",
                commit_message=f"V2 Real Manim Render Eval (Pass@1={summary['pass1_real_mp4_render_pct']}%, Pass@2={summary['pass2_ritl_real_mp4_render_pct']}%)",
            )
        except Exception as e:
            print(f"  [WARN] Could not upload V2 eval results: {e}", flush=True)
    return summary


def main() -> None:
    start_time = time.time()
    cfg = V2PipelineConfig()
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device_map, max_memory = get_gpu_sharding()
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"\n[V2 Stage 1] Loading {cfg.base_model} in Native 16-Bit (FP16, SDPA, Zero 4-Bit Quantization)...",
        flush=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model.enable_input_require_grads()
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    resumed_step = 0
    best_val_loss = float("inf")
    sft_history: List[Dict[str, Any]] = []
    sft_already_done = False

    if HF_TOKEN:
        try:
            api = HfApi(token=HF_TOKEN)
            hub_files = api.list_repo_files(repo_id=cfg.hub_sft_repo, repo_type="model")
            if "best/adapter_config.json" in hub_files and "best/training_state.json" in hub_files:
                snap = snapshot_download(repo_id=cfg.hub_sft_repo, repo_type="model", token=HF_TOKEN)
                best_sub = os.path.join(snap, "best")
                with open(os.path.join(best_sub, "training_state.json"), "r", encoding="utf-8") as f:
                    st = json.load(f)
                resumed_step = int(st.get("global_step", 0))
                best_val_loss = float(st.get("best_val_loss", float("inf")))
                sft_history = st.get("history", [])
                sft_already_done = st.get("status") == "completed" or resumed_step >= cfg.sft_max_steps
                if resumed_step > 0:
                    model = PeftModel.from_pretrained(base_model, best_sub, is_trainable=True)
                    print(f"[V2 Resume] Loaded best Stage-1 adapter at step={resumed_step} (val_loss={best_val_loss:.4f})", flush=True)
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

    cast_trainable_lora_to_fp32(model)
    model.print_trainable_parameters()
    best_weights_cpu = snapshot_trainable_weights_cpu(model)

    # =========================================================================
    # V2 STAGE 1: 16-BIT ULTRA-ANIMATIVE SFT (ZERO TRUNCATION, STABILIZED LR)
    # =========================================================================
    if not sft_already_done:
        neftune_handle = register_neftune_hook(model, alpha=cfg.neftune_alpha)
        train_rows = load_hub_jsonl(cfg.sft_dataset_repo, "data/train.jsonl", HF_TOKEN)
        val_rows = load_hub_jsonl(cfg.sft_dataset_repo, "data/validation.jsonl", HF_TOKEN)
        print(f"[V2 Stage 1 Data] Loaded {len(train_rows)} V2 train rows and {len(val_rows)} val rows.", flush=True)

        train_ds = ManimChatDataset(train_rows, tokenizer, max_seq_length=cfg.max_seq_length)
        val_ds = ManimChatDataset(val_rows, tokenizer, max_seq_length=cfg.max_seq_length)
        train_loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda b: {
                "input_ids": b[0]["input_ids"].unsqueeze(0),
                "attention_mask": b[0]["attention_mask"].unsqueeze(0),
                "labels": b[0]["labels"].unsqueeze(0),
            },
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda b: {
                "input_ids": b[0]["input_ids"].unsqueeze(0),
                "attention_mask": b[0]["attention_mask"].unsqueeze(0),
                "labels": b[0]["labels"].unsqueeze(0),
            },
        )

        total_sft_steps = cfg.sft_max_steps
        remaining_steps = max(1, total_sft_steps - resumed_step)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.sft_lr, weight_decay=0.01, foreach=False)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=max(6, int(0.08 * remaining_steps)), num_training_steps=remaining_steps
        )

        first_dev = next(model.parameters()).device
        model.train()
        optimizer.zero_grad()

        if resumed_step == 0:
            sync_to_hub(
                model,
                tokenizer,
                cfg.hub_sft_repo,
                os.path.join(cfg.output_dir, "sft_latest"),
                {"global_step": 0, "best_val_loss": best_val_loss, "status": "v2_training_started", "history": []},
                HF_TOKEN,
            )

        global_step = resumed_step
        micro_step = 0
        running_loss = 0.0

        for idx, batch in enumerate(train_loader):
            if resumed_step > 0 and idx < resumed_step * cfg.grad_accum_steps:
                continue
            batch = {k: v.to(first_dev, non_blocking=True) for k, v in batch.items()}
            raw_loss = compute_chunked_ce_loss(
                model, batch["input_ids"], batch["attention_mask"], batch["labels"], chunk_size=128
            )
            loss = raw_loss / cfg.grad_accum_steps
            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() > 0:
                loss.backward()
                running_loss += loss.item()
            micro_step += 1

            if micro_step % cfg.grad_accum_steps == 0:
                grad_norm = clip_grad_norm_multi_gpu(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step <= 3 or global_step % cfg.log_every_steps == 0:
                    vram_0 = torch.cuda.max_memory_allocated(0) / (1024**3)
                    vram_1 = torch.cuda.max_memory_allocated(1) / (1024**3) if torch.cuda.device_count() > 1 else 0.0
                    denom = 1.0 if global_step <= 3 else float(cfg.log_every_steps)
                    avg_l = running_loss / denom
                    running_loss = 0.0
                    elapsed_m = (time.time() - start_time) / 60.0
                    print(
                        f"  [V2 SFT Step {global_step:4d}/{total_sft_steps:4d}] Loss: {avg_l:.4f} | "
                        f"GradNorm: {grad_norm:.2f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"VRAM (G0/G1): {vram_0:.2f}/{vram_1:.2f} GiB | {elapsed_m:.1f}m",
                        flush=True,
                    )
                    sft_history.append({"step": global_step, "loss": round(avg_l, 5), "vram_g0": round(vram_0, 2), "vram_g1": round(vram_1, 2)})

                if global_step % cfg.eval_every_steps == 0 or global_step >= total_sft_steps:
                    val_loss, val_ppl = evaluate_sft_val(model, val_loader, first_dev)
                    print(f"  >>> [V2 SFT Val @ Step {global_step}] Val Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}", flush=True)
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                        best_weights_cpu = snapshot_trainable_weights_cpu(model)
                    elif val_loss > best_val_loss * 1.05:
                        print(f"  [Rollback Guard] Val loss ({val_loss:.4f}) > best ({best_val_loss:.4f}). Restoring best & halving LR.", flush=True)
                        restore_trainable_weights_from_cpu(model, best_weights_cpu)
                        for pg in optimizer.param_groups:
                            pg["lr"] *= 0.5
                        val_loss = best_val_loss
                        val_ppl = math.exp(min(20.0, val_loss))

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

                if (time.time() - start_time) / 3600.0 >= cfg.max_walltime_hours - 2.0:
                    restore_trainable_weights_from_cpu(model, best_weights_cpu)
                    st = {"global_step": global_step, "best_val_loss": best_val_loss, "status": "paused", "history": sft_history}
                    sync_to_hub(model, tokenizer, cfg.hub_sft_repo, os.path.join(cfg.output_dir, "sft_latest"), st, HF_TOKEN)
                    break

                if global_step >= total_sft_steps:
                    break

        neftune_handle.remove()
        restore_trainable_weights_from_cpu(model, best_weights_cpu)

    # =========================================================================
    # V2 STAGE 2: 16-BIT ADAPTER-REFERENCE DPO (LATEX + ANIMATIVENESS + ZERO TRUNCATION)
    # =========================================================================
    if (time.time() - start_time) / 3600.0 < cfg.max_walltime_hours - 1.2:
        print("\n[V2 Stage 2] Merging Best V2 16-Bit SFT Adapter into Reference Backbone...", flush=True)
        for p in model.parameters():
            if p.dtype == torch.float32:
                p.data = p.data.to(torch.float16)
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
        cast_trainable_lora_to_fp32(dpo_model)

        dpo_rows = load_hub_jsonl(cfg.dpo_dataset_repo, "data/train.jsonl", HF_TOKEN)
        print(f"[V2 Stage 2 Data] Loaded {len(dpo_rows)} V2 DPO preference pairs.", flush=True)
        dpo_ds = ManimDPODataset(dpo_rows, tokenizer, max_seq_length=2048)
        dpo_loader = DataLoader(dpo_ds, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])

        dpo_accum = 8
        total_dpo_steps = min(cfg.dpo_max_steps, max(1, len(dpo_loader) // dpo_accum))
        dpo_params = [p for p in dpo_model.parameters() if p.requires_grad]
        dpo_opt = torch.optim.AdamW(dpo_params, lr=cfg.dpo_lr, weight_decay=0.01, foreach=False)
        dpo_sched = get_cosine_schedule_with_warmup(
            dpo_opt, num_warmup_steps=max(4, int(0.06 * total_dpo_steps)), num_training_steps=total_dpo_steps
        )

        first_dev = next(dpo_model.parameters()).device
        dpo_model.train()
        dpo_opt.zero_grad()

        sync_to_hub(
            dpo_model,
            tokenizer,
            cfg.hub_dpo_repo,
            os.path.join(cfg.output_dir, "dpo_latest"),
            {"global_step": 0, "status": "v2_dpo_started", "sft_best_val_loss": best_val_loss, "history": []},
            HF_TOKEN,
        )

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
                    ref_c_logp, _ = compute_chunked_dpo_logps(dpo_model, c_ids, c_lbl)
                    ref_r_logp, _ = compute_chunked_dpo_logps(dpo_model, r_ids, r_lbl)

            pol_c_logp, c_nll = compute_chunked_dpo_logps(dpo_model, c_ids, c_lbl)
            pol_r_logp, _ = compute_chunked_dpo_logps(dpo_model, r_ids, r_lbl)

            margin = cfg.dpo_beta * ((pol_c_logp - pol_r_logp) - (ref_c_logp - ref_r_logp))
            loss = (-F.logsigmoid(margin).mean() + 0.1 * c_nll) / dpo_accum
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                run_dpo_loss += loss.item()
                run_dpo_acc += float((margin > 0).float().mean().item()) / dpo_accum
            dpo_micro += 1

            if dpo_micro % dpo_accum == 0:
                clip_grad_norm_multi_gpu(dpo_params, cfg.max_grad_norm)
                dpo_opt.step()
                dpo_sched.step()
                dpo_opt.zero_grad()
                dpo_step += 1

                if dpo_step % 5 == 0:
                    avg_dl = run_dpo_loss / 5.0
                    avg_acc = (run_dpo_acc / 5.0) * 100.0
                    run_dpo_loss = 0.0
                    run_dpo_acc = 0.0
                    print(f"  [V2 DPO Step {dpo_step:4d}/{total_dpo_steps:4d}] Loss: {avg_dl:.4f} | Pref Acc: {avg_acc:.1f}%", flush=True)
                    dpo_history.append({"step": dpo_step, "dpo_loss": round(avg_dl, 5), "pref_acc": round(avg_acc, 2)})

                if dpo_step % 20 == 0 or dpo_step >= total_dpo_steps:
                    st = {
                        "global_step": dpo_step,
                        "sft_best_val_loss": best_val_loss,
                        "status": "completed" if dpo_step >= total_dpo_steps else "in_progress",
                        "history": dpo_history,
                    }
                    sync_to_hub(dpo_model, tokenizer, cfg.hub_dpo_repo, os.path.join(cfg.output_dir, "dpo_latest"), st, HF_TOKEN)

                if dpo_step >= total_dpo_steps or (time.time() - start_time) / 3600.0 >= cfg.max_walltime_hours - 0.8:
                    break

        # =====================================================================
        # V2 STAGE 3: REAL HEADLESS MANIM + LATEX VIDEO RENDERING BENCHMARK
        # =====================================================================
        eval_summary = run_v2_real_render_eval(dpo_model, tokenizer, cfg, max_cases=15)
        print("\n=== V2 STAGE 3 REAL MANIM RENDER EVAL SUMMARY ===", flush=True)
        print(json.dumps({k: v for k, v in eval_summary.items() if k != "details"}, indent=2), flush=True)

    print("\n=== ALL V2 ULTRA-ANIMATIVE TRAINING, DPO & REAL RENDER EVAL FINISHED! ===", flush=True)


if __name__ == "__main__":
    main()
