# FinetuningExps — `v2.0-sft-16bit-engine`: Pure 16-Bit Dual-GPU High-Rank `rsLoRA` + `NEFTune` SFT Engine

> **Branch Version:** `v2.0-sft-16bit-engine`  
> **Base Backbone:** `Qwen/Qwen2.5-Coder-7B-Instruct` (Native `torch.float16`, Zero 4-Bit Quantization)  
> **Target Hardware:** Kaggle `GPU T4 x2` (`2 × 16 GB = 32 GB VRAM`)  
> **Live Checkpoint Repository:** [`PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit`](https://huggingface.co/PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit)

---

## 1. Why Pure 16-Bit (`FP16`) Dual-GPU Sharding Beats 4-Bit Unsloth

While 4-bit QLoRA (`NF4`) is popular on single GPUs, compressing weights to 4-bit introduces quantization noise in mathematical coordinate calculations and causes an activation-to-weight mismatch when merging adapters back into 16-bit weights. Furthermore, single-GPU tools leave Kaggle's second T4 GPU completely idle.

Our custom PyTorch SFT engine (`src/training/sft_trainer.py`) uses **zero 4-bit quantization**:

1. **Balanced Native 16-Bit Pipeline Sharding (`device_map="auto"`):**
   - Shards `Qwen/Qwen2.5-Coder-7B-Instruct` ($7.6\text{B} \times 2\text{ bytes} = 15.2\text{ GB}$) evenly across `cuda:0` ($\approx 7.6\text{ GB}$) and `cuda:1` ($\approx 7.6\text{ GB}$) using `SDPA` (Scaled Dot-Product Attention) and non-reentrant gradient checkpointing.
2. **High-Rank Rank-Stabilized LoRA (`rsLoRA`):**
   - Trains $r = 64, \alpha = 64$ across all 7 linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with Rank-Stabilized scaling:
     $$\Delta W = \frac{\alpha}{\sqrt{r}} B A = \frac{64}{\sqrt{64}} B A = 8.0 \cdot B A$$
   - Prevents gradient collapse at high rank and matches full 16-bit fine-tuning capacity.
3. **Exact Multi-Turn Completion-Only Loss Masking (`ManimChatDataset`):**
   - Tokenizes full multi-turn ChatML conversations (`system`, `user`, `assistant`) and masks every `system` and `user` token with `-100`:
     $$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{|T_{\text{assistant}}|} \sum_{t \in T_{\text{assistant}}} \log P_\theta(y_t \mid y_{<t}, x)$$
   - Ensures 100% of parameter updates go toward learning `<think>` spatial layout planning, ManimCE v0.19+ syntax, and multi-turn compiler self-repair.
4. **Custom `NEFTune` Embedding Noise Hook ($\alpha = 5.0$):**
   - Injects uniform noise into token embeddings during training forward passes:
     $$\tilde{\mathbf{e}} = \mathbf{e} + \mathcal{U}(-1, 1) \cdot \frac{\alpha}{\sqrt{L \cdot d}}$$
   - Empirically improves code instruction adherence and reduces syntactic hallucinations.
5. **Fault-Tolerant HuggingFace Hub Checkpointing & Auto-Resume:**
   - Automatically syncs adapter weights, tokenizer, validation perplexity, and `training_state.json` to `PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit` every `save_every_steps`, and automatically resumes from the latest Hub step across Kaggle sessions.

---

## 2. Running Stage-1 16-Bit SFT

```bash
export HF_TOKEN="your_hf_token"
python src/training/sft_trainer.py \
  --base_model Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset_repo PRB1/ManimCE-Master-SFT \
  --hub_model_repo PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit \
  --lora_r 64 \
  --lora_alpha 64 \
  --max_seq_length 3072 \
  --learning_rate 1.5e-4 \
  --num_epochs 2
```

