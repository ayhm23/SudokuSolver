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

# Order a camera applies them in: geometry (how it is held), optics (focus),
# sensor (noise), then compression when the file is saved.
ORDER = ("rotate", "blur", "gaussian_noise", "jpeg")


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
        out = gaussian_noise(out, s, seed) if name == "gaussian_noise" else globals()[name](out, s)
    return out
