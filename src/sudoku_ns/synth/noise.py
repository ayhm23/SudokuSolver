"""Image degradations with a single severity knob s in [0, 1].

Each function maps s to one physical parameter (documented below), so a
noise sweep can say "blur severity 0.5 = Gaussian sigma 1.5 px" in the report.
s = 0 is always the identity (or, for JPEG, near-lossless quality 95).

| function        | parameter at severity s          | s = 1 means                      |
|-----------------|----------------------------------|----------------------------------|
| blur            | Gaussian sigma = 3.0 * s px      | strong defocus / motion smear    |
| rotate          | angle = 15 * s degrees (CCW)     | photo held noticeably crooked    |
| jpeg            | quality = round(95 - 85 * s)     | quality 10, heavy block artefacts |
| gaussian_noise  | sigma = 25 * s (0-255 units)     | dim-light phone sensor noise     |
| lighting        | darkest point = 1 - 0.7 * s of   | strong side light + a shadow     |
|                 | the original brightness          | blob (30% brightness left)       |
| clutter         | 12 * s distractor objects        | busy desk: frames, boxes, lines, |
|                 | drawn around (never on) the grid | text, a fake small grid          |

clutter needs the grid corners (it must not cover the grid, which would be
occlusion, a different problem), so it is not part of apply_noise.

Images are uint8 (H, W) or (H, W, 3); every function returns the same shape
and dtype.
"""

from __future__ import annotations

import cv2
import numpy as np

BLUR_MAX_SIGMA = 3.0
ROTATE_MAX_DEG = 15.0
JPEG_Q_HI, JPEG_Q_LO = 95, 10
NOISE_MAX_SIGMA = 25.0
LIGHT_MAX_DROP = 0.7
CLUTTER_MAX_OBJECTS = 12

# Order a camera applies them in: geometry (how it is held), optics (focus),
# sensor (noise), then compression when the file is saved.
# Lighting is a property of the scene, so it comes before everything else.
ORDER = ("lighting", "rotate", "blur", "gaussian_noise", "jpeg")


def _check(s: float) -> float:
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {s}")
    return float(s)


def blur(img: np.ndarray, s: float) -> np.ndarray:
    sigma = BLUR_MAX_SIGMA * _check(s)
    if sigma == 0:
        return img.copy()
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def rotation_matrix(shape: tuple[int, ...], s: float) -> np.ndarray:
    """2x3 affine matrix used by rotate(); apply it to corners with cv2.transform."""
    h, w = shape[:2]
    return cv2.getRotationMatrix2D((w / 2, h / 2), ROTATE_MAX_DEG * _check(s), 1.0)


def rotate(img: np.ndarray, s: float) -> np.ndarray:
    """Rotate about the centre; borders are replicated so no black wedges appear."""
    if _check(s) == 0:
        return img.copy()
    h, w = img.shape[:2]
    return cv2.warpAffine(
        img, rotation_matrix(img.shape, s), (w, h), borderMode=cv2.BORDER_REPLICATE
    )


def jpeg(img: np.ndarray, s: float) -> np.ndarray:
    quality = round(JPEG_Q_HI - (JPEG_Q_HI - JPEG_Q_LO) * _check(s))
    # cv2 expects BGR; swapping channels in and out keeps RGB input correct.
    bgr = img[..., ::-1] if img.ndim == 3 else img
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:  # pragma: no cover
        raise RuntimeError("JPEG encoding failed")
    out = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return out[..., ::-1].copy() if img.ndim == 3 else out


def gaussian_noise(img: np.ndarray, s: float, seed: int | None = None) -> np.ndarray:
    sigma = NOISE_MAX_SIGMA * _check(s)
    if sigma == 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def lighting(img: np.ndarray, s: float, seed: int | None = None) -> np.ndarray:
    """Uneven illumination: a linear fall-off in a random direction times a soft shadow.

    This is what breaks a single global threshold (one side of the grid ends up
    darker than the ink on the other side), which is why detection uses an
    adaptive threshold and CLAHE.
    """
    drop = LIGHT_MAX_DROP * _check(s)
    if drop == 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = rng.uniform(0, 2 * np.pi)
    t = np.cos(a) * xx / w + np.sin(a) * yy / h
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    gain = 1 - drop * t * 0.6  # gradient takes up to 60% of the drop ...
    blob = np.zeros((h, w), np.float32)
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    axes = (int(rng.uniform(0.15, 0.35) * w), int(rng.uniform(0.15, 0.35) * h))
    cv2.ellipse(blob, (int(cx), int(cy)), axes, rng.uniform(0, 180), 0, 360, 1.0, -1)
    blob = cv2.GaussianBlur(blob, (0, 0), 0.08 * min(h, w))
    gain *= 1 - drop * 0.4 * blob / max(blob.max(), 1e-6)  # ... a shadow blob the rest
    gain = gain[..., None] if img.ndim == 3 else gain
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def clutter(img: np.ndarray, corners: np.ndarray, s: float, seed: int | None = None) -> np.ndarray:
    """Draw distractors on the background around the grid (never on it).

    Objects: rectangle outlines (some ENCLOSING the grid, i.e. larger contours
    than the grid itself), filled boxes, long lines, digit text and a small fake
    grid. They target the detector's two assumptions: "the grid is the
    biggest quadrilateral" and "anything with lines inside is the grid".
    """
    n = round(CLUTTER_MAX_OBJECTS * _check(s))
    if n == 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    layer = img.copy()
    pts = np.asarray(corners, np.float32)
    lo, hi = pts.min(axis=0), pts.max(axis=0)

    def colour():
        v = int(rng.integers(0, 256))
        return (v, v, v) if img.ndim == 2 else tuple(int(c) for c in rng.integers(0, 256, 3))

    for i in range(n):
        kind = ("frame", "box", "line", "text", "fakegrid")[i % 5]
        if kind == "frame":  # enclosing outline, 5-40 px outside the grid
            pad = rng.uniform(5, 40, size=2)
            p0 = np.maximum(lo - pad, 0).astype(int)
            p1 = np.minimum(hi + pad, [w - 1, h - 1]).astype(int)
            cv2.rectangle(layer, tuple(p0), tuple(p1), colour(), int(rng.integers(2, 6)))
        elif kind == "box":
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            bw, bh = rng.integers(20, max(21, w // 3), size=2)
            cv2.rectangle(layer, (x, y), (int(x + bw), int(y + bh)), colour(), -1)
        elif kind == "line":
            p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(layer, p0, p1, colour(), int(rng.integers(1, 5)))
        elif kind == "text":
            org = (int(rng.integers(0, w)), int(rng.integers(20, h)))
            txt = "".join(str(d) for d in rng.integers(0, 10, size=int(rng.integers(2, 7))))
            cv2.putText(
                layer,
                txt,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                rng.uniform(0.6, 2.0),
                colour(),
                int(rng.integers(1, 4)),
            )
        else:  # fake 4x4 grid of lines
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            step = int(rng.integers(8, 25))
            for k in range(5):
                cv2.line(layer, (x, y + k * step), (x + 4 * step, y + k * step), colour(), 1)
                cv2.line(layer, (x + k * step, y), (x + k * step, y + 4 * step), colour(), 1)
    # Restore the grid region (dilated a little so the border stays intact).
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, pts.astype(np.int32), 255)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0
    layer[mask] = img[mask]
    return layer


def noisy_corners(
    shape: tuple[int, ...], corners: np.ndarray, spec: dict[str, float]
) -> np.ndarray:
    """Where the grid corners end up after apply_noise(img, spec): only rotation moves them."""
    corners = np.asarray(corners, np.float32)
    if spec.get("rotate", 0) == 0:
        return corners.copy()
    return cv2.transform(corners[None], rotation_matrix(shape, spec["rotate"]))[0]


def random_spec(
    rng: np.random.Generator, types: tuple[str, ...] = ORDER, max_types: int = 2
) -> dict[str, float]:
    """1..max_types distinct noise types, each at a uniform severity in [0.1, 1]."""
    k = int(rng.integers(1, max_types + 1))
    names = rng.choice(types, size=k, replace=False)
    return {str(n): round(float(rng.uniform(0.1, 1.0)), 3) for n in names}


def apply_noise(img: np.ndarray, spec: dict[str, float], seed: int | None = None) -> np.ndarray:
    """Apply several degradations, e.g. {"blur": 0.3, "jpeg": 0.5}, in camera ORDER."""
    unknown = set(spec) - set(ORDER)
    if unknown:
        raise ValueError(f"unknown noise type(s) {sorted(unknown)}; choose from {ORDER}")
    out = img
    for name in ORDER:
        if name not in spec:
            continue
        s = spec[name]
        if name in ("gaussian_noise", "lighting"):
            out = globals()[name](out, s, seed)
        else:
            out = globals()[name](out, s)
    return out
