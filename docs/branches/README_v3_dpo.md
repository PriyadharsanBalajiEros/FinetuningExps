# FinetuningExps — `v3.0-dpo-compiler-alignment`: Zero-Overhead 16-Bit Adapter-Reference DPO + RPO-NLL Alignment

> **Branch Version:** `v3.0-dpo-compiler-alignment`  
> **Stage-1 Reference Model ($\pi_{\text{ref}}$):** `PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit` (Merged in-memory in 16-bit)  
> **Stage-2 Preference Dataset:** [`PRB1/ManimCE-Master-DPO`](https://huggingface.co/datasets/PRB1/ManimCE-Master-DPO)  
> **Stage-2 Aligned Model Repo:** [`PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit`](https://huggingface.co/PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit)

---

## 1. Zero-Memory-Duplication 16-Bit Reference Policy

Standard Direct Preference Optimization (DPO) loads two separate copies of a 7B model ($\pi_\theta$ and $\pi_{\text{ref}}$), requiring $>30\text{ GB}$ of VRAM for weights alone in 16-bit precision.

Our custom DPO engine (`src/training/dpo_trainer.py`) eliminates duplicate reference model VRAM completely:

1. **In-Memory 16-Bit Stage-1 Merge (`merge_and_unload()`):**
   - Loads `Qwen/Qwen2.5-Coder-7B-Instruct` in `torch.float16` across both T4 GPUs and merges the Stage-1 `rsLoRA` SFT adapter directly into the 16-bit base weights.
   - The frozen 16-bit backbone now **is** the exact Stage-1 SFT reference policy $\pi_{\text{ref}}$.
2. **Adapter-Toggled Reference Forward Pass (`with model.disable_adapter():`):**
   - Attaches a fresh Stage-2 16-bit `rsLoRA` adapter ($r = 32, \alpha = 32$).
   - During each training step, computing $\log \pi_{\text{ref}}(y_w \mid x)$ and $\log \pi_{\text{ref}}(y_l \mid x)$ inside `with torch.no_grad(): with model.disable_adapter():` uses **0.00 bytes of extra VRAM**.
3. **RPO / CPO NLL-Stabilized DPO Objective:**
   - In code generation,chosen ($y_w$) and rejected ($y_l$) scripts share boilerplate Python tokens (`from manim import *`, `class Scene`, `def construct(self):`). Pure DPO can inadvertently depress log-probabilities of valid Python syntax.
   - We optimize the RPO-stabilized objective:
     $$\mathcal{L}_{\text{Total}} = -\log \sigma \left( \beta \left[ \left(\log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)\right) - \left(\log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x)\right) \right] \right) + \lambda_{\text{NLL}} \mathcal{L}_{\text{NLL}}(y_w \mid x)$$
     with $\beta = 0.1$ and $\lambda_{\text{NLL}} = 0.1$.

---

## 2. Targeted Failure Modes Eliminated by Stage-2 DPO

| Rejection Category | Chosen ($y_w$) | Rejected ($y_l$) |
| :--- | :--- | :--- |
| **`legacy_api_hallucination`** | `Create(...)`, `MathTex(...)`, `axes.plot(...)` | `ShowCreation(...)`, `TexMobject(...)`, `axes.get_graph(...)` |
| **`invalid_axes_kwarg`** | `Axes(x_range=[-3, 3], axis_config={"color": BLUE})` | `Axes(x_range=[-3, 3], axis_color=BLUE)` (`TypeError`) |
| **`wrong_3d_scene_base`** | `class SurfacePlot(ThreeDScene):` with `set_camera_orientation` | `class SurfacePlot(Scene):` (`AttributeError`) |
| **`spatial_layout_overlap`** | Explicit `.to_edge(UP)`, `.next_to(..., DOWN, buff=0.4)` | Unpositioned Mobjects stacked on top of `ORIGIN` |

---

## 3. Running Stage-2 16-Bit DPO Alignment

```bash
export HF_TOKEN="your_hf_token"
python src/training/dpo_trainer.py \
  --base_model Qwen/Qwen2.5-Coder-7B-Instruct \
  --sft_adapter_repo PRB1/Qwen2.5-Coder-7B-ManimCE-SFT-16bit \
  --dpo_dataset_repo PRB1/ManimCE-Master-DPO \
  --hub_dpo_repo PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit
```
