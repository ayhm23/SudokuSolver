"""Interactive demo: photo -> grid -> CNN beliefs -> MaxSAT, next to the argmax baseline.

    pip install -e ".[ml,demo]"
    streamlit run demo/app.py

Everything the system computes comes from sudoku_ns.pipeline.run (vision + CNN +
one reasoning method) and sudoku_ns.pipeline.visualize (the 3-panel figure). This
file only chooses the input, calls those two functions, and lays out what they
return. Explanation text comes from sudoku_ns.reasoning.explain, unchanged.

Inputs: an uploaded photo, or a test image (synthetic test_final/clean, or a real
wichtounet v2 test photo) with optional synthetic noise (synth/noise.py). Test images
are read as raw pixels, exactly like the evaluation (scripts/run_e2e.py), so their
ground truth applies; uploads go through pipeline.run's EXIF rotation as usual.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from sudoku_ns.interfaces import SolveStatus
from sudoku_ns.pipeline import run, visualize
from sudoku_ns.reasoning.explain import describe_override, format_grid
from sudoku_ns.synth.noise import ORDER, apply_noise
from sudoku_ns.synth.render import load_render

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "outputs/models/cnn_v1/model.pt"
SETS = {
    "synthetic (test_final/clean)": ROOT / "data/synthetic/test_final/clean",
    "real photo (wichtounet v2 test)": ROOT / "data/real/wichtounet_v2_test",
}
TAU = 0.5  # threshold_then_sat tau of the final runs (not used by the methods shown here)
UNIQUE_LIMIT_S = 10
NOISE_SEED = 0
TMP = Path(tempfile.gettempdir()) / "sudoku_ns_demo"


@st.cache_resource
def perceiver():
    from sudoku_ns.perception.perceiver import CNNPerceiver

    return CNNPerceiver(CHECKPOINT)


@st.cache_data(show_spinner=False)
def test_images(folder: str) -> list[str]:
    return [p.stem for p in sorted(Path(folder).glob("*.json"))]


def solve(path: Path, method: str, solver_kwargs: dict | None = None):
    return run(path, perceiver(), method=method, tau=TAU, solver_kwargs=solver_kwargs)


def pick_input() -> tuple[Path, np.ndarray, dict | None, str] | None:
    """Sidebar. Returns (image file for pipeline.run, pixels shown, ground truth or None, label)."""
    st.sidebar.header("Input")
    source = st.sidebar.radio("source", ["test image", "upload a photo"])
    TMP.mkdir(exist_ok=True)
    if source == "upload a photo":
        up = st.sidebar.file_uploader("photo of a Sudoku", type=["jpg", "jpeg", "png"])
        if up is None:
            st.info("Upload a photo of a printed Sudoku, or pick a test image in the sidebar.")
            return None
        path = TMP / f"upload{Path(up.name).suffix.lower()}"
        path.write_bytes(up.getvalue())
        try:
            shown = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
        except (UnidentifiedImageError, OSError):
            st.error("That file could not be read as an image. Try a JPEG or PNG photo.")
            return None
        return path, shown, None, up.name

    set_name = st.sidebar.selectbox("test set", list(SETS))
    folder = SETS[set_name]
    ids = test_images(str(folder))
    if not ids:
        st.error(f"No test images in {folder}. See README (Reproduce everything) to create them.")
        return None
    image_id = st.sidebar.selectbox("image", ids)
    noise_type = st.sidebar.selectbox("noise", ["none", *ORDER])
    severity = 0.0
    if noise_type != "none":
        severity = st.sidebar.slider("severity", 0.0, 1.0, 0.5, 0.05)
    r = load_render(folder / f"{image_id}.json")  # raw pixels, like the evaluation
    spec = {noise_type: severity} if noise_type != "none" and severity > 0 else {}
    pixels = apply_noise(r.image, spec, seed=NOISE_SEED) if spec else r.image
    path = TMP / "test_input.png"  # PNG: lossless, no EXIF, so pipeline.run sees these pixels
    Image.fromarray(pixels).save(path)
    label = f"{folder.name}/{image_id}" + (f"  +  {json.dumps(spec)}" if spec else "")
    return path, pixels, {"givens": r.givens, "solution": r.solution}, label


def confidence_heatmap(prob: np.ndarray):
    import matplotlib.pyplot as plt

    conf = prob.max(2)
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    im = ax.imshow(conf, cmap="magma", vmin=0, vmax=1)
    for r in range(9):
        for c in range(9):
            ax.text(c, r, f"{conf[r, c]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if conf[r, c] < 0.6 else "black")  # fmt: skip
    for i in (2.5, 5.5):
        ax.axhline(i, color="white", lw=1.5)
        ax.axvline(i, color="white", lw=1.5)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, label="top-class probability")
    ax.set_title("CNN confidence per cell", fontsize=9)
    fig.tight_layout()
    return fig


def verdict(res, truth: dict | None) -> str:
    if res.status is not SolveStatus.SOLVED:
        return f"**{res.status.value.upper()}**"
    if truth is None:
        return "**solved** (no ground truth for uploads)"
    ok = np.array_equal(res.solution, truth["solution"])
    return "**solved, correct** ✅" if ok else "**solved, WRONG** ❌"


def method_column(col, title: str, out, truth: dict | None, note: str = "") -> None:
    res = out.result
    with col:
        st.subheader(title)
        st.markdown(verdict(res, truth))
        a, b = st.columns(2)
        a.metric("solve time", f"{res.solve_time_s * 1000:.0f} ms")
        b.metric("unique solution", "-" if res.unique is None else ("yes" if res.unique else "NO"))
        if note:
            st.warning(note)
        if res.message and res.status is not SolveStatus.SOLVED:
            st.caption(res.message)
        if res.overrides:
            st.markdown(f"**{len(res.overrides)} override(s)** (cell = (row, col), 0-based):")
            for o in res.overrides:
                st.markdown(f"- `{describe_override(o)}`")
        elif res.status is SolveStatus.SOLVED:
            st.caption("no overrides: the solver kept every CNN reading")
        if res.solution is not None:
            over = {(o.row, o.col) for o in res.overrides}
            st.code(format_grid(res.solution, over, color=False), language=None)
            st.caption("[d] = a cell where this method overrode the CNN")


def main() -> None:
    st.set_page_config(page_title="Neurosymbolic Sudoku", layout="wide")
    st.title("Neurosymbolic Sudoku: CNN beliefs + weighted MaxSAT")
    if not CHECKPOINT.exists():
        st.error(f"CNN checkpoint not found at {CHECKPOINT}. Train it first (README, Week 2).")
        return
    picked = pick_input()
    if picked is None:
        return
    path, pixels, truth, label = picked
    st.sidebar.header("Methods")
    want_unique = st.sidebar.checkbox(
        f"also run maxsat_unique ({UNIQUE_LIMIT_S} s limit)", value=False
    )
    st.caption(f"input: {label}")

    try:
        with st.spinner("vision + CNN + maxsat_weighted ..."):
            main_out = solve(path, "maxsat_weighted")
    except Exception as e:  # never show a stack trace in the demo
        st.error(f"The pipeline failed on this input: {type(e).__name__}: {e}")
        return

    if main_out.result.status is SolveStatus.ERROR:
        st.error(
            "**The grid could not be found in this image**, so nothing was read or solved.\n\n"
            f"Reason given by the vision stage: `{main_out.result.message}`\n\n"
            "Try a photo where the whole grid is visible, roughly square-on and in focus."
        )
        st.image(pixels, caption="input", width=420)
        return

    vis = TMP / "vis.png"
    visualize(pixels, main_out, vis, f"{label} - maxsat_weighted")
    st.image(str(vis), width="stretch")
    st.caption(
        "Left: detected grid. Middle: CNN top class and probability per cell (black >= 0.9, "
        "orange >= 0.6, red < 0.6). Right: maxsat_weighted solution (black = read givens, "
        "red = overridden by the solver, blue = filled in)."
    )

    left, right = st.columns([1, 2])
    with left:
        st.pyplot(confidence_heatmap(main_out.prob))
        if truth is not None:
            wrong = int((main_out.prob.argmax(2) != truth["givens"]).sum())
            st.metric("CNN argmax cells wrong vs ground truth", f"{wrong} / 81")
    with right:
        st.markdown("**CNN argmax reading** (what a pipeline without reasoning would solve):")
        st.code(format_grid(main_out.prob.argmax(2), color=False), language=None)

    with st.spinner("argmax_then_sat ..."):
        base_out = solve(path, "argmax_then_sat")
    cols = st.columns(3 if want_unique else 2)
    method_column(cols[0], "maxsat_weighted (ours)", main_out, truth)
    method_column(cols[1], "argmax_then_sat (baseline)", base_out, truth)
    if want_unique:
        with st.spinner(f"maxsat_unique (up to ~{UNIQUE_LIMIT_S} s) ..."):
            uniq_out = solve(path, "maxsat_unique", {"time_limit": UNIQUE_LIMIT_S})
        msg = uniq_out.result.message or ""
        note = (
            f"timed out after {UNIQUE_LIMIT_S} s, showing weighted answer" if
            msg.startswith("timeout") else ""
        )  # fmt: skip
        method_column(cols[2], "maxsat_unique", uniq_out, truth, note)


main()
