"""The public-dataset importer: .dat parsing and EXIF corner conversion."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from import_wichtounet import exif_to_raw, parse_dat, unit_repeats  # noqa: E402

DAT = """sonyEricsson s500i
640x480:24 JPG
0 0 0 7 0 0 0 8 0
0 9 0 0 0 3 1 0 0
0 0 6 8 0 5 0 7 0
0 2 0 6 0 0 0 4 9
0 0 0 2 0 0 0 5 0
0 0 8 0 4 0 0 0 7
0 0 0 9 0 0 0 3 0
3 7 0 0 0 0 0 0 6
1 0 5 0 0 4 0 0 0
"""


def test_parse_dat(tmp_path):
    p = tmp_path / "x.dat"
    p.write_text(DAT, encoding="utf-8")
    phone, res, g = parse_dat(p)
    assert phone == "sonyEricsson s500i" and res == "640x480"
    assert g.shape == (9, 9) and g[0, 3] == 7 and g[8, 0] == 1 and (g > 0).sum() == 26
    assert unit_repeats(g) == []
    g[0, 0] = 7
    assert unit_repeats(g) == ["row 1"]


def test_parse_dat_rejects_short_grid(tmp_path):
    p = tmp_path / "x.dat"
    p.write_text("\n".join(DAT.splitlines()[:-1]), encoding="utf-8")
    with pytest.raises(ValueError):
        parse_dat(p)


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_exif_to_raw_matches_pil_transpose(orientation):
    # A raw image with one marked pixel; PIL's exif_transpose gives the displayed
    # image. Mapping the marked pixel's displayed position back must hit it.
    from PIL import Image, ImageOps

    raw = np.zeros((40, 60), np.uint8)  # h=40, w=60
    raw[7, 11] = 255
    im = Image.fromarray(raw)
    exif = im.getexif()
    exif[274] = orientation
    im.info["exif"] = exif.tobytes()
    shown = np.asarray(ImageOps.exif_transpose(im))
    yd, xd = np.argwhere(shown == 255)[0]
    back = exif_to_raw(np.array([[xd, yd]], np.float32), orientation, 60, 40)
    np.testing.assert_allclose(back, [[11, 7]])
