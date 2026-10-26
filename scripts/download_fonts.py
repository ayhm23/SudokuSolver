"""Download OFL-licensed printed fonts from the google/fonts GitHub repo into data/fonts/.

    python scripts/download_fonts.py

Why: the synthetic renderer draws each puzzle in a random font, so the Week 2
CNN learns digit shapes rather than one typeface. The dataset holds out whole
fonts for testing, so we need a dozen or so, spread across styles. Fonts are
gitignored (data/) and each family's OFL.txt licence is saved alongside.

File names are resolved through the GitHub API at download time (families
switch between static and variable files over time), so nothing is hardcoded
except the family directory and a preferred-file hint.
"""

import argparse
import json
import urllib.request
from pathlib import Path

API = "https://api.github.com/repos/google/fonts/contents/ofl/{family}"

# (family directory, style, hint that picks one file from that directory)
FONTS = [
    ("merriweather", "serif", "Merriweather"),
    ("lora", "serif", "Lora"),
    ("librebaskerville", "serif", "LibreBaskerville"),
    ("ptserif", "serif", "PT_Serif-Web-Regular"),
    ("lato", "sans", "Lato-Regular"),
    ("firasans", "sans", "FiraSans-Regular"),
    ("opensans", "sans", "OpenSans"),
    ("spacemono", "mono", "SpaceMono-Regular"),
    ("ibmplexmono", "mono", "IBMPlexMono-Regular"),
    ("sourcecodepro", "mono", "SourceCodePro"),
    ("barlowcondensed", "condensed", "BarlowCondensed-Regular"),
    ("oswald", "condensed", "Oswald"),
    ("firasans", "bold", "FiraSans-Bold"),
    ("lato", "bold", "Lato-Black"),
    ("ptserif", "bold", "PT_Serif-Web-Bold"),
]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sudoku-ns-font-fetch"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def pick(files: list[dict], hint: str) -> dict | None:
    # Prefer an exact stem match ("Lato-Regular.ttf"), then any .ttf starting
    # with the hint (variable fonts like "Lora[wght].ttf"), never italics.
    ttfs = [f for f in files if f["name"].endswith(".ttf") and "Italic" not in f["name"]]
    for f in ttfs:
        if f["name"] == f"{hint}.ttf":
            return f
    starts = [f for f in ttfs if f["name"].startswith(hint)]
    return min(starts, key=lambda f: len(f["name"]), default=None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/fonts"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest, listings = [], {}
    for family, style, hint in FONTS:
        if family not in listings:
            listings[family] = json.loads(fetch(API.format(family=family)))
        f = pick(listings[family], hint)
        if f is None:
            print(f"SKIP {family}: no .ttf matching {hint!r}")
            continue
        dest = args.out / f["name"]
        if not dest.exists():
            dest.write_bytes(fetch(f["download_url"]))
        lic = args.out / f"{family}_OFL.txt"
        if not lic.exists():
            lic_file = next((x for x in listings[family] if x["name"] == "OFL.txt"), None)
            if lic_file:
                lic.write_bytes(fetch(lic_file["download_url"]))
        manifest.append({"file": f["name"], "family": family, "style": style})
        print(f"ok   {style:<9} {f['name']}")

    (args.out / "fonts.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} fonts in {args.out}")


if __name__ == "__main__":
    main()
