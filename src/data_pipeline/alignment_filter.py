"""
Prompt-to-Code Alignment Verifier & Contradiction Detector.
===========================================================
Catches row-shuffled / misaligned (prompt, code) pairs found in several
public HuggingFace Manim datasets (such as derivatives of `generaleoley/manim-codegen`)
by cross-checking domain concepts, geometric primitives, mathematical symbols,
and explicit user string literals between the prompt and the Python AST.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


# Domain concept groups mapping prompt keywords -> expected code tokens / Mobjects
CONCEPT_GROUPS: Dict[str, Tuple[Set[str], Set[str]]] = {
    "3d_geometry": (
        {"3d", "three-dimensional", "three dimensional", "cylinder", "sphere", "cube", "cone", "torus", "surface"},
        {"ThreeDScene", "ThreeDAxes", "Cylinder", "Sphere", "Cube", "Cone", "Torus", "Surface", "set_camera_orientation", "DEGREES"},
    ),
    "calculus_integral": (
        {"integral", "integrate", "integration", "area under", "riemann", "antiderivative", "definite integral"},
        {"int", "get_area", "get_riemann_rectangles", "Axes", "plot", "MathTex", "Integral"},
    ),
    "calculus_derivative": (
        {"derivative", "differentiate", "tangent line", "slope", "secant", "rate of change"},
        {"secant", "tangent", "ValueTracker", "always_redraw", "Axes", "plot", "frac", "d/dx", "dx"},
    ),
    "linear_algebra": (
        {"matrix", "matrices", "eigenvector", "eigenvalue", "determinant", "linear transformation", "svd", "singular value", "vector"},
        {"Matrix", "Vector", "Arrow", "NumberPlane", "LinearTransformationScene", "apply_matrix", "MathTex", "array", "bmatrix", "pmatrix"},
    ),
    "trigonometry": (
        {"sine", "cosine", "tangent", "sin(x)", "cos(x)", "unit circle", "fourier", "wave", "oscillat", "harmonic"},
        {"sin", "cos", "tan", "PI", "TAU", "Circle", "Axes", "plot", "ValueTracker"},
    ),
    "complex_numbers": (
        {"complex plane", "euler's formula", "euler formula", "imaginary", "mandelbrot", "roots of unity"},
        {"ComplexPlane", "e^{i", "exp", "complex", "MathTex", "PI"},
    ),
    "probability_stats": (
        {"probability", "normal distribution", "gaussian", "bell curve", "histogram", "bar chart", "bayes", "monte carlo", "random"},
        {"BarChart", "random", "exp", "Axes", "Rectangle", "Dot", "SampleSpace"},
    ),
    "graph_theory": (
        {"graph theory", "vertices", "edges", "shortest path", "dijkstra", "bfs", "dfs", "binary tree", "neural network"},
        {"Graph", "DiGraph", "Line", "Arrow", "Circle", "Dot", "VGroup", "vertices", "edges"},
    ),
}

STOPWORDS: Set[str] = {
    "the", "and", "for", "that", "this", "with", "from", "your", "have", "create",
    "make", "write", "manim", "manimce", "python", "code", "scene", "animation",
    "animated", "video", "using", "show", "showing", "display", "displaying",
    "would", "like", "please", "could", "can", "you", "want", "user", "request",
    "generate", "accurate", "correct", "requested", "here", "educational",
    "step", "first", "then", "finally", "screen", "object", "objects",
}


@dataclass
class AlignmentScore:
    is_aligned: bool
    score: float
    matched_concepts: List[str]
    reason: str


def _split_identifier(text: str) -> Set[str]:
    """Split CamelCase and snake_case identifiers into lowercase tokens."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    spaced = spaced.replace("_", " ")
    tokens = re.findall(r"[a-zA-Z]{3,}", spaced.lower())
    return {t for t in tokens if t not in STOPWORDS}


def verify_prompt_code_alignment(
    prompt: str,
    code: str,
    is_trusted_source: bool = False,
) -> AlignmentScore:
    """
    Verify whether `prompt` and `code` are genuinely aligned or come from
    a row-shuffled dataset.
    """
    if not prompt or not code:
        return AlignmentScore(False, 0.0, [], "empty_prompt_or_code")

    prompt_lower = prompt.lower()
    code_lower = code.lower()

    # 1. Explicit quoted proper-noun contradiction check
    # (e.g., Prompt asks for `"Kiron"` and `"Tisha"`, but code doesn't contain them)
    quoted_names = re.findall(r'["\']([A-Z][a-zA-Z]{3,15})["\']', prompt)
    for qname in quoted_names:
        if qname.lower() not in STOPWORDS and qname.lower() not in {"manim", "scene", "math", "text", "blue", "red", "green", "yellow", "white"}:
            if qname.lower() not in code_lower:
                return AlignmentScore(
                    False,
                    0.0,
                    [],
                    f"missing_explicit_quoted_literal:{qname}",
                )

    # 2. Check Strong Contradictions (e.g. Prompt explicitly asks for a 3D cylinder/sphere,
    #    or a 2x2 Matrix, while code is an unrelated integral or text flip)
    if any(w in prompt_lower for w in ["cylinder", "3d coordinate", "three-dimensional", "3d scene", "sphere"]):
        if not any(tok in code for tok in ["ThreeD", "Cylinder", "Sphere", "Cube", "Surface", "z_range", "phi", "theta", "3D", "3d"]):
            return AlignmentScore(False, 0.0, [], "contradiction:prompt_3d_but_code_2d")

    if "matrix" in prompt_lower and ("[[" in prompt_lower or "2x2" in prompt_lower or "3x3" in prompt_lower):
        if not any(tok in code for tok in ["Matrix", "matrix", "bmatrix", "pmatrix", "array", "LinearTransformation"]):
            return AlignmentScore(False, 0.0, [], "contradiction:prompt_matrix_but_code_has_no_matrix")

    if "lissajous" in prompt_lower and "lissajous" not in code_lower and "sin" not in code_lower:
        return AlignmentScore(False, 0.0, [], "contradiction:prompt_lissajous_mismatch")

    # 3. Domain Concept Group Matching
    matched_groups: List[str] = []
    contradicted_groups: List[str] = []
    for group_name, (prompt_triggers, code_Anchor_tokens) in CONCEPT_GROUPS.items():
        if any(trig in prompt_lower for trig in prompt_triggers):
            if any(anchor.lower() in code_lower for anchor in code_Anchor_tokens):
                matched_groups.append(group_name)
            else:
                contradicted_groups.append(group_name)

    # 4. Lexical overlap between prompt keywords and code identifiers/comments/strings
    prompt_words = _split_identifier(prompt)
    code_words = _split_identifier(code)
    lexical_overlap = prompt_words.intersection(code_words)

    # Compute composite alignment score
    score = 0.25 * len(matched_groups) + 0.08 * len(lexical_overlap) - 0.35 * len(contradicted_groups)

    if is_trusted_source:
        # Trusted sources only fail on explicit contradictions above
        return AlignmentScore(True, max(score, 0.5), matched_groups + list(lexical_overlap)[:5], "trusted_verified")

    # For untrusted/noisy sources, require strong positive lexical or concept match
    # and zero unresolved domain contradictions
    if contradicted_groups and not matched_groups:
        return AlignmentScore(
            False,
            score,
            matched_groups,
            f"unmatched_domain_concepts:{','.join(contradicted_groups)}",
        )

    if len(matched_groups) >= 1 or len(lexical_overlap) >= 2:
        return AlignmentScore(
            True,
            score,
            matched_groups + list(lexical_overlap)[:6],
            "aligned_verified",
        )

    return AlignmentScore(False, score, [], "insufficient_prompt_code_overlap")

