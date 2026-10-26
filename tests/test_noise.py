import numpy as np
import pytest

from sudoku_ns.synth.noise import apply_noise, blur, gaussian_noise, jpeg, rotate


@pytest.fixture
def img():
    # Grid-like test pattern: sharp edges make every degradation measurable.
    a = np.full((96, 96, 3), 240, np.uint8)
    a[::12] = 20
    a[:, ::12] = 20
    return a


def mse(a, b):
    return float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())


@pytest.mark.parametrize("fn", [blur, rotate, gaussian_noise])
def test_zero_severity_is_identity(fn, img):
    np.testing.assert_array_equal(fn(img, 0.0), img)


def test_jpeg_zero_severity_is_near_lossless(img):
    assert mse(jpeg(img, 0.0), img) < mse(jpeg(img, 1.0), img)
    assert mse(jpeg(img, 0.0), img) < 50


@pytest.mark.parametrize("fn", [blur, rotate, jpeg, gaussian_noise])
def test_damage_grows_with_severity(fn, img):
    errs = [mse(fn(img, s), img) for s in (0.25, 0.5, 1.0)]
    assert errs == sorted(errs) and errs[0] > 0


@pytest.mark.parametrize("fn", [blur, rotate, jpeg, gaussian_noise])
def test_shape_and_dtype_preserved(fn, img):
    for x in (img, img[..., 0].copy()):
        out = fn(x, 0.7)
        assert out.shape == x.shape and out.dtype == np.uint8


@pytest.mark.parametrize("fn", [blur, rotate, jpeg, gaussian_noise])
def test_rejects_bad_severity(fn, img):
    with pytest.raises(ValueError, match="severity"):
        fn(img, 1.5)


def test_apply_noise_composes_and_validates(img):
    out = apply_noise(img, {"blur": 0.3, "jpeg": 0.5, "gaussian_noise": 0.2}, seed=0)
    assert out.shape == img.shape and mse(out, img) > 0
    np.testing.assert_array_equal(
        out, apply_noise(img, {"blur": 0.3, "jpeg": 0.5, "gaussian_noise": 0.2}, seed=0)
    )
    with pytest.raises(ValueError, match="unknown"):
        apply_noise(img, {"fog": 0.5})
