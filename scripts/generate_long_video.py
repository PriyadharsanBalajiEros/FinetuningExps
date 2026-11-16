"""
Long-Form (5 to 10 Minute) Multi-Act ManimCE Documentary Generator & FFmpeg Stitcher.
=====================================================================================
Stress-tests our 16-bit SFT + DPO model (`PRB1/Qwen2.5-Coder-7B-ManimCE-DPO-16bit`)
on long-form 3Blue1Brown-style mathematical documentaries (5 to 10 minutes) by:
  1. Decomposing a deep mathematical topic into 6–8 structured Pedagogical Acts
     (45–75 seconds per Act, 8–12 animation beats per Act with explicit `run_time`
     and `self.wait()` pacing).
  2. Generating, AST-canonicalizing, and rendering each Act independently with
     1-turn Renderer-in-the-Loop (RITL) compiler self-repair.
  3. Stitching all rendered Act `.mp4` videos losslessly via `ffmpeg -f concat`
     into a single continuous 5–10 minute master `.mp4` video.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.infer_manim import generate_and_render_manim, load_manimce_16bit_model


DEFAULT_FOURIER_LAPLACE_ACTS: List[Tuple[str, str]] = [
    (
        "Act 1 — The Mystery of Pure Waves & Superposition",
        "Create a long, deeply detailed 60-second ManimCE animation (at least 10 distinct self.play / self.wait stages, "
        "using run_time=2.5 to 3.5 and self.wait(2) between stages) introducing wave superposition. "
        "Start with a title 'Act I: The Language of Waves' at the top edge (.to_edge(UP)), draw a coordinate Axes, "
        "plot three pure sine waves y1 = sin(x), y2 = 0.5*sin(3x), and y3 = 0.33*sin(5x) in BLUE, TEAL, and GREEN, "
        "show their equations with MathTex, and smoothly transform them into their combined sum wave in YELLOW "
        "showing how a square wave emerges. Clean up intermediate labels with FadeOut to prevent overlap.",
    ),
    (
        "Act 2 — Winding a Wave Around the Complex Circle",
        "Create a long, deeply detailed 60-second ManimCE animation (at least 10 distinct self.play / self.wait stages, "
        "using run_time=2.5 to 3.5 and self.wait(2)) demonstrating 3Blue1Brown's core Fourier intuition: winding a graph "
        "around the origin in the complex plane using Euler's formula g(t) * e^{-2*pi*i*f*t}. "
        "Show a ComplexPlane on the left and a time-domain Axes on the right (or sequentially with FadeOut), plot a "
        "polar wound curve for different winding frequencies, highlight the center-of-mass Dot in RED, and display "
        "the step-by-step MathTex derivation of Euler's winding integral.",
    ),
    (
        "Act 3 — The Center of Mass Spike & Frequency Spectrum",
        "Create a long, deeply detailed 60-second ManimCE animation (at least 10 distinct self.play / self.wait stages, "
        "using run_time=2.5 to 3.5 and self.wait(2)) visualizing how the center of mass of the wound curve stays near "
        "the origin for most frequencies but spikes dramatically when the winding frequency f matches the signal frequency! "
        "Plot the frequency spectrum |F(f)| on an Axes, animate a ValueTracker sweeping across frequencies, trace the spike "
        "at f = 1 and f = 3, and annotate the peaks with Arrows and MathTex formulas.",
    ),
    (
        "Act 4 — Rotating Epicycles Reconstructing Complex Shapes",
        "Create a long, deeply detailed 60-second ManimCE animation (at least 10 distinct self.play / self.wait stages, "
        "using run_time=2.5 to 3.5 and self.wait(2)) showing a chain of rotating Fourier epicycle circles and Arrows "
        "(fundamental + 3rd + 5th + 7th harmonics) tip-to-tail, tracing out a sharp waveform over time. "
        "Display the Fourier series summation formula sum_{k} c_k e^{i k omega t} at the top edge and highlight each "
        "harmonic's amplitude circle.",
    ),
    (
        "Act 5 — Heisenberg Uncertainty & Time-Frequency Tradeoff",
        "Create a long, deeply detailed 60-second ManimCE animation (at least 10 distinct self.play / self.wait stages, "
        "using run_time=2.5 to 3.5 and self.wait(2)) illustrating the Time-Frequency Uncertainty Principle: when a Gaussian "
        "wave packet in time is compressed narrower, its Fourier transform Gaussian in frequency stretches wider! "
        "Plot both the time-domain wave packet and frequency-domain spectrum using Axes and ValueTracker, and derive "
        "Delta_t * Delta_omega >= 1/2 using step-by-step MathTex equations.",
    ),
    (
        "Act 6 — From Fourier to Laplace: Complex s-Plane & Grand Synthesis",
        "Create a long, deeply detailed 60-second ManimCE animation (at least 10 distinct self.play / self.wait stages, "
        "using run_time=2.5 to 3.5 and self.wait(2)) extending the Fourier transform into the Laplace transform F(s) = "
        "int_0^infty f(t) e^{-s t} dt where s = sigma + i*omega. Show how the imaginary axis i*omega is the Fourier "
        "transform, illustrate damped oscillations e^{-sigma t} cos(omega t) converging, and conclude with a grand "
        "summary comparison table of Time Domain vs Frequency Domain.",
    ),
]


def stitch_mp4_videos(mp4_paths: List[str], output_master_mp4: str = "master_documentary.mp4") -> Optional[str]:
    """Concatenate rendered Act `.mp4` files losslessly using `ffmpeg -f concat`."""
    valid_paths = [os.path.abspath(p) for p in mp4_paths if p and os.path.exists(p)]
    if not valid_paths:
        return None
    concat_txt = "ffmpeg_concat_list.txt"
    with open(concat_txt, "w", encoding="utf-8") as f:
        for vp in valid_paths:
            escaped = vp.replace("\\", "/")
            f.write(f"file '{escaped}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", output_master_mp4]
    print(f"\n[FFmpeg Stitcher] Stitching {len(valid_paths)} Acts into {output_master_mp4} ...")
    subprocess.run(cmd, check=False)
    return output_master_mp4 if os.path.exists(output_master_mp4) else None


def build_long_documentary(
    acts: List[Tuple[str, str]] = DEFAULT_FOURIER_LAPLACE_ACTS,
    quality: str = "m",
    output_master_mp4: str = "fourier_to_laplace_masterclass.mp4",
) -> Optional[str]:
    model, tokenizer = load_manimce_16bit_model(hf_token=os.environ.get("HF_TOKEN"))
    rendered_mp4s: List[str] = []

    for idx, (act_title, act_prompt) in enumerate(acts, start=1):
        print("=" * 78)
        print(f"Generating & Rendering [{idx}/{len(acts)}]: {act_title}")
        print("=" * 78)
        py_out = f"act_{idx:02d}_scene.py"
        mp4 = generate_and_render_manim(
            model=model,
            tokenizer=tokenizer,
            prompt=act_prompt,
            max_new_tokens=3072,
            output_py_path=py_out,
            render_quality=quality,
            media_dir=f"media_act_{idx:02d}",
        )
        if mp4:
            rendered_mp4s.append(mp4)

    return stitch_mp4_videos(rendered_mp4s, output_master_mp4=output_master_mp4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 5-10 minute multi-act ManimCE documentary")
    parser.add_argument("--quality", type=str, default="m", choices=["l", "m", "h"])
    parser.add_argument("--output", type=str, default="fourier_to_laplace_masterclass.mp4")
    args = parser.parse_args()
    build_long_documentary(quality=args.quality, output_master_mp4=args.output)
