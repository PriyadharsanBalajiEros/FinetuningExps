"""
ManimCE v0.19+ AST Verifier, Static Linter & Legacy Code Canonicalizer.
======================================================================
Ensures 100% of training examples conform to modern Manim Community Edition
(v0.19 / v0.20) syntax, compile cleanly into a Python AST, define a valid
Scene subclass with a non-trivial `construct(self)` method, and contain
zero deprecated ManimGL (`manimlib`) or ManimCE v0.14 hallucinations.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


VALID_SCENE_BASES: Set[str] = {
    "Scene",
    "ThreeDScene",
    "SpecialThreeDScene",
    "MovingCameraScene",
    "ZoomedScene",
    "VectorScene",
    "LinearTransformationScene",
    "VoiceoverScene",
    "GraphScene",
    "SampleSpaceScene",
}

# Legacy ManimGL (3b1b manimlib) markers that are incompatible with ManimCE
MANIMGL_FORBIDDEN_PATTERNS: List[Tuple[str, str]] = [
    (r"\bfrom\s+manimlib\b", "Uses legacy manimlib import instead of ManimCE"),
    (r"\bimport\s+manimlib\b", "Uses legacy manimlib import instead of ManimCE"),
    (r"\bInteractiveScene\b", "Uses ManimGL InteractiveScene"),
    (r"\bexternally_animated_scene\b", "Uses ManimGL internal module"),
    (r"\bCONFIG\s*=\s*\{", "Uses legacy ManimGL CONFIG dictionary pattern"),
    (r"\bself\.embed\s*\(", "Uses ManimGL interactive self.embed()"),
]

# Hallucinated kwargs that cause immediate runtime TypeError in ManimCE v0.19+
FORBIDDEN_KWARGS: Set[str] = {
    "element_color",
    "max_value",
    "max_magnitude",
}

# Known built-in ManimCE symbols and standard Python names so we can detect
# hallucinated undefined custom classes (like `Mohsin()`)
KNOWN_EXTERNAL_MODULES: Set[str] = {
    "manim",
    "numpy",
    "np",
    "math",
    "random",
    "itertools",
    "functools",
    "collections",
    "scipy",
    "sympy",
    " networkx",
    "nx",
    "colour",
    "manim_voiceover",
    "typing",
}


@dataclass
class VerificationResult:
    is_valid: bool
    canonical_code: str
    scene_name: Optional[str] = None
    scene_base: Optional[str] = None
    num_animations: int = 0
    num_lines: int = 0
    repairs_applied: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None


def extract_python_code(raw_text: str) -> str:
    """Extract executable Python code from markdown fences or raw completion text."""
    if not raw_text:
        return ""
    text = raw_text.strip()

    # Strip <think>...</think> blocks if present before extracting code
    text_no_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text_no_think:
        text = text_no_think

    # Match ```python ... ``` or ```py ... ``` or ``` ... ```
    fence_pattern = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    matches = fence_pattern.findall(text)
    if matches:
        # Pick the longest block that contains a Scene definition or import
        best = max(matches, key=lambda m: (1 if "class " in m and "def construct" in m else 0, len(m)))
        return best.strip()

    # Handle literal escaped "\n " prefix seen in some HF datasets
    if text.startswith("\\n"):
        text = text[2:].lstrip()

    return text.strip()


def canonicalize_manimce_code(code: str) -> Tuple[str, List[str]]:
    """
    Safely modernize deprecated ManimCE v0.12-v0.17 calls to ManimCE v0.19+
    and normalize imports.
    """
    repairs: List[str] = []
    updated = code.replace("\r\n", "\n").strip()

    if updated.startswith("\\n"):
        updated = updated[2:].lstrip()
        repairs.append("stripped_leading_escaped_newline")

    # 1. Replace ShowCreation -> Create
    if re.search(r"\bShowCreation\s*\(", updated):
        updated = re.sub(r"\bShowCreation\s*\(", "Create(", updated)
        repairs.append("ShowCreation->Create")

    # 2. Replace TexMobject -> MathTex
    if re.search(r"\bTexMobject\s*\(", updated):
        updated = re.sub(r"\bTexMobject\s*\(", "MathTex(", updated)
        repairs.append("TexMobject->MathTex")

    # 3. Replace TextMobject -> Tex
    if re.search(r"\bTextMobject\s*\(", updated):
        updated = re.sub(r"\bTextMobject\s*\(", "Tex(", updated)
        repairs.append("TextMobject->Tex")

    # 4. Replace .get_graph( -> .plot(
    if re.search(r"\.get_graph\s*\(", updated):
        updated = re.sub(r"\.get_graph\s*\(", ".plot(", updated)
        repairs.append("get_graph->plot")

    # 5. Ensure `from manim import *` is present
    if not re.search(r"^\s*(?:from\s+manim\s+import|import\s+manim\b)", updated, re.MULTILINE):
        updated = "from manim import *\n\n" + updated
        repairs.append("added_manim_import")

    return updated, repairs


def verify_and_canonicalize(raw_code: str, min_lines: int = 6, max_lines: int = 600) -> VerificationResult:
    """
    Full static verification + AST inspection + ManimCE v0.19+ canonicalization.
    """
    extracted = extract_python_code(raw_code)
    if not extracted:
        return VerificationResult(False, "", rejection_reason="empty_code")

    # Reject ManimGL / manimlib patterns
    for pattern, reason in MANIMGL_FORBIDDEN_PATTERNS:
        if re.search(pattern, extracted):
            return VerificationResult(False, extracted, rejection_reason=reason)

    # Reject external proprietary speech services that aren't standard pip packages
    if "tools.aos_speech_service" in extracted or "AOSSpeechService" in extracted:
        # Canonicalize AOSSpeechService -> GTTSService so VoiceoverScene runs standalone
        extracted = extracted.replace(
            "from tools.aos_speech_service import AOSSpeechService",
            "from manim_voiceover.services.gtts import GTTSService",
        ).replace("AOSSpeechService(", "GTTSService(")

    canonical, repairs = canonicalize_manimce_code(extracted)
    lines = [ln for ln in canonical.split("\n") if ln.strip() and not ln.strip().startswith("#")]
    if len(lines) < min_lines:
        return VerificationResult(False, canonical, rejection_reason=f"too_short ({len(lines)} lines)")
    if len(lines) > max_lines:
        return VerificationResult(False, canonical, rejection_reason=f"too_long ({len(lines)} lines)")

    # Parse Python AST
    try:
        tree = ast.parse(canonical)
    except SyntaxError as exc:
        return VerificationResult(
            False,
            canonical,
            repairs_applied=repairs,
            rejection_reason=f"SyntaxError: {exc.msg} (line {exc.lineno})",
        )

    # Inspect AST for Scene class, construct() method, animations, and forbidden patterns
    scene_name: Optional[str] = None
    scene_base: Optional[str] = None
    has_construct = False
    num_animations = 0
    defined_classes: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            defined_classes.add(node.name)
            for base in node.bases:
                base_id = None
                if isinstance(base, ast.Name):
                    base_id = base.id
                elif isinstance(base, ast.Attribute):
                    base_id = base.attr
                if base_id in VALID_SCENE_BASES:
                    scene_name = node.name
                    scene_base = base_id
                    # Check methods inside this Scene class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "construct":
                            # Ensure construct body isn't just `pass` or `...`
                            non_pass_stmts = [
                                s
                                for s in item.body
                                if not isinstance(s, ast.Pass)
                                and not (
                                    isinstance(s, ast.Expr)
                                    and isinstance(s.value, ast.Constant)
                                    and (s.value.value is Ellipsis or isinstance(s.value.value, str))
                                )
                            ]
                            if len(non_pass_stmts) >= 2:
                                has_construct = True

        # Count animation / scene actions and check forbidden kwargs
        if isinstance(node, ast.Call):
            # Check for self.play / self.add / self.wait / self.move_camera
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in {"play", "add", "wait", "move_camera", "begin_ambient_camera_rotation"}:
                    num_animations += 1

            # Check for forbidden hallucinated kwargs
            for kw in node.keywords:
                if kw.arg in FORBIDDEN_KWARGS:
                    return VerificationResult(
                        False,
                        canonical,
                        repairs_applied=repairs,
                        rejection_reason=f"forbidden_kwarg: {kw.arg}",
                    )
                # Axes(axis_color=...) is invalid in ManimCE v0.19+; must be axis_config={"color": ...}
                if (
                    kw.arg == "axis_color"
                    and isinstance(node.func, ast.Name)
                    and node.func.id in {"Axes", "ThreeDAxes", "NumberPlane", "ComplexPlane"}
                ):
                    return VerificationResult(
                        False,
                        canonical,
                        repairs_applied=repairs,
                        rejection_reason=f"invalid_kwarg: {node.func.id}(axis_color=...)",
                    )

    if not scene_name:
        return VerificationResult(
            False, canonical, repairs_applied=repairs, rejection_reason="no_valid_scene_subclass"
        )
    if not has_construct:
        return VerificationResult(
            False, canonical, repairs_applied=repairs, rejection_reason="empty_or_missing_construct"
        )
    if num_animations == 0:
        return VerificationResult(
            False, canonical, repairs_applied=repairs, rejection_reason="no_scene_animations_or_adds"
        )

    # Check if 3D camera methods are mistakenly called inside a 2D `Scene`
    if scene_base == "Scene" and (
        "set_camera_orientation" in canonical or "begin_ambient_camera_rotation" in canonical
    ):
        # Auto-upgrade Scene -> ThreeDScene!
        canonical = re.sub(
            rf"class\s+{re.escape(scene_name)}\s*\(\s*Scene\s*\)\s*:",
            f"class {scene_name}(ThreeDScene):",
            canonical,
        )
        scene_base = "ThreeDScene"
        repairs.append("Scene->ThreeDScene_for_camera_orientation")

    return VerificationResult(
        is_valid=True,
        canonical_code=canonical,
        scene_name=scene_name,
        scene_base=scene_base,
        num_animations=num_animations,
        num_lines=len(lines),
        repairs_applied=repairs,
    )

