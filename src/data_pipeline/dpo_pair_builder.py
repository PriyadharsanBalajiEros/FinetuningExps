"""
Stage 2 DPO Preference Pair & Renderer-in-the-Loop (RITL) Self-Repair Builder.
==============================================================================
Constructs:
1. Hard-Negative Direct Preference Optimization (DPO) pairs `(prompt, chosen, rejected)`
   targeting the exact failure modes of generalist LLMs (like Claude Opus / GPT-4o):
   - Legacy ManimGL / v0.14 API hallucinations (`ShowCreation`, `TexMobject`, `get_graph`)
   - Invalid ManimCE v0.19+ constructor kwargs (`Axes(axis_color=...)`, `element_color=...`)
   - 3D camera calls inside 2D `Scene` (`AttributeError: 'Scene' object has no attribute 'set_camera_orientation'`)
   - Spatial layout overlap (stacking all Mobjects at `ORIGIN` without `.to_edge()` / `.next_to()`)
2. Multi-Turn Renderer-in-the-Loop (RITL) Compiler Traceback Self-Repair trajectories
   so the model learns to fix any Manim runtime exception in 1 turn.
"""

import random
import re
from typing import Dict, List, Optional, Tuple


SYSTEM_PROMPT_MANIMCE = (
    "You are an expert Manim Community Edition (ManimCE v0.19+) Python developer and "
    "mathematical animator. Given a user request, first plan the spatial layout, coordinate "
    "bounds, timing, and pedagogical flow inside <think>...</think> tags (when reasoning is helpful), "
    "and then output a complete, runnable, self-contained ManimCE Python script using `from manim import *`. "
    "Never use deprecated ManimGL (`manimlib`) methods like `ShowCreation`, `TexMobject`, `TextMobject`, "
    "or `axes.get_graph`; always use modern ManimCE APIs (`Create`, `MathTex`, `Tex`, `Text`, `axes.plot`) "
    "and prevent visual overlap using `.to_edge()`, `.next_to()`, and `FadeOut`."
)


def synthesize_hard_negative_code(
    clean_code: str,
    seed: int = 42,
) -> Optional[Tuple[str, str, str]]:
    """
    Given a verified ManimCE v0.19+ script (`chosen`), inject a realistic LLM
    hallucination or spatial bug to produce `(rejected_code, bug_category, simulated_traceback)`.
    """
    rng = random.Random(seed)
    candidates: List[Tuple[str, str, str]] = []

    # Perturbation 1: Legacy ManimGL / ManimCE v0.14 API Hallucination
    if "Create(" in clean_code or "MathTex(" in clean_code or ".plot(" in clean_code:
        corrupted = clean_code
        trace_msg = ""
        if "Create(" in corrupted:
            corrupted = corrupted.replace("Create(", "ShowCreation(", 1)
            trace_msg = "NameError: name 'ShowCreation' is not defined. Did you mean: 'Create'?"
        if "MathTex(" in corrupted:
            corrupted = corrupted.replace("MathTex(", "TexMobject(", 1)
            trace_msg = trace_msg or "NameError: name 'TexMobject' is not defined. In ManimCE, use 'MathTex'."
        if ".plot(" in corrupted:
            corrupted = corrupted.replace(".plot(", ".get_graph(", 1)
            trace_msg = trace_msg or "AttributeError: 'Axes' object has no attribute 'get_graph'. Did you mean: 'plot'?"
        if corrupted != clean_code:
            candidates.append((corrupted, "legacy_api_hallucination", trace_msg))

    # Perturbation 2: Invalid Axes constructor keyword argument (`axis_color=BLUE`)
    if "Axes(" in clean_code and "axis_color=" not in clean_code:
        corrupted = clean_code.replace("Axes(", "Axes(axis_color=BLUE, ", 1)
        trace_msg = (
            "TypeError: Mobject.__init__() got an unexpected keyword argument 'axis_color'. "
            "Pass axis styling via `axis_config={'color': BLUE}` instead."
        )
        candidates.append((corrupted, "invalid_axes_kwarg", trace_msg))

    # Perturbation 3: Calling 3D camera orientation on 2D `Scene`
    if "(ThreeDScene):" in clean_code and "set_camera_orientation" in clean_code:
        corrupted = clean_code.replace("(ThreeDScene):", "(Scene):", 1)
        trace_msg = (
            "AttributeError: 'Scene' object has no attribute 'set_camera_orientation'. "
            "Inherit from `ThreeDScene` when using 3D camera orientation."
        )
        candidates.append((corrupted, "wrong_3d_scene_base", trace_msg))

    # Perturbation 4: Spatial Layout Overlap (removing positioning calls `.to_edge`, `.next_to`, `.shift`)
    layout_pattern = re.compile(r"\.(?:to_edge|next_to|to_corner|shift)\s*\([^()\n]*\)")
    if len(layout_pattern.findall(clean_code)) >= 3:
        corrupted = layout_pattern.sub("", clean_code, count=3)
        if corrupted != clean_code:
            trace_msg = (
                "VisualLayoutWarning: Multiple Text/MathTex Mobjects are rendered at ORIGIN (0, 0, 0) "
                "causing severe visual overlap. Use `.to_edge(UP)`, `.next_to(..., DOWN, buff=0.4)`, or `.shift()`."
            )
            candidates.append((corrupted, "spatial_layout_overlap", trace_msg))

    if not candidates:
        return None

    return rng.choice(candidates)


def build_dpo_and_ritl_records(
    verified_sft_records: List[Dict],
    max_synthetic_dpo: int = 2200,
    max_ritl_repairs: int = 1200,
) -> Tuple[List[Dict], List[Dict]]:
    """
    From verified Tier-1 SFT records, generate:
    1. `dpo_records`: `(system, prompt, chosen, rejected, category, source)`
    2. `ritl_sft_records`: Multi-turn compiler self-repair conversations for SFT
    """
    dpo_records: List[Dict] = []
    ritl_records: List[Dict] = []

    # Prioritize gold / high-complexity scenes for synthesizing hard negatives
    eligible = [
        r for r in verified_sft_records
        if r.get("tier", 2) == 1 and len(r.get("code", "")) >= 350
    ]
    if len(eligible) < max_synthetic_dpo:
        eligible = verified_sft_records

    for idx, rec in enumerate(eligible):
        if len(dpo_records) >= max_synthetic_dpo and len(ritl_records) >= max_ritl_repairs:
            break

        clean_code = rec["code"]
        prompt = rec["prompt"]
        res = synthesize_hard_negative_code(clean_code, seed=1000 + idx)
        if not res:
            continue

        rejected_code, bug_category, simulated_traceback = res
        chosen_formatted = f"```python\n{clean_code}\n```"
        rejected_formatted = f"```python\n{rejected_code}\n```"

        if len(dpo_records) < max_synthetic_dpo:
            dpo_records.append(
                {
                    "system": SYSTEM_PROMPT_MANIMCE,
                    "prompt": prompt,
                    "chosen": chosen_formatted,
                    "rejected": rejected_formatted,
                    "category": bug_category,
                    "source": f"synthetic_hard_neg:{rec.get('source', 'gold')}",
                }
            )

        # Build Renderer-in-the-Loop (RITL) 2-turn self-repair SFT trajectory
        if len(ritl_records) < max_ritl_repairs and idx % 2 == 0:
            repair_user_msg = (
                "Running the generated Manim script produced the following compiler/layout error:\n\n"
                f"```text\n{simulated_traceback}\n```\n\n"
                "Please fix the bug and provide the complete corrected ManimCE v0.19+ Python script."
            )
            ritl_records.append(
                {
                    "id": f"ritl_repair_{len(ritl_records):05d}",
                    "source": f"ritl_self_repair:{bug_category}",
                    "tier": 1,
                    "scene_name": rec.get("scene_name", "MainScene"),
                    "has_reasoning": False,
                    "prompt": prompt,
                    "code": clean_code,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_MANIMCE},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": rejected_formatted},
                        {"role": "user", "content": repair_user_msg},
                        {"role": "assistant", "content": chosen_formatted},
                    ],
                }
            )

    return dpo_records, ritl_records

