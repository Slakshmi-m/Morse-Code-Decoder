"""
Download a curated set of Morse Code Ninja practice packs for model training.

Downloads ~30 ZIP files covering:
  - Single Letters at 4 speeds            (A-Z coverage)
  - Single Letter-Number at 4 speeds      (digits 0-9 coverage)
  - Top 500 Words at 4 speeds × 2 spacing (sequence decoding + Farnsworth variety)

Usage:
  python download_ninja.py                        # downloads to data/ninja_raw/
  python download_ninja.py --out data/ninja_raw
  python download_ninja.py --dry-run              # print URLs without downloading

After this, run:
  python scripts/prepare_real_data.py data/ninja_raw/ --output-dir data/real
"""

import argparse
import os
import urllib.error
import urllib.parse
import urllib.request
import zipfile

CDN = ("https://podcasts-morsecode-ninja.nyc3.cdn.digitaloceanspaces.com"
       "/practice/files/individual-files/")

# ── curated download list ────────────────────────────────────────────────────
# Format: (display name, URL-encoded filename)
# Single Letters — alphabet coverage at 4 speeds
SINGLE_LETTERS = [
    f"Single Letters {w}wpm" for w in [15, 20, 25, 35]
]

# Single Letter-Number — digit coverage at 4 speeds
SINGLE_LETTER_NUMBER = [
    f"Single Letter-Number {w}wpm" for w in [15, 20, 25, 35]
]

# Top 500 Words with Farnsworth-style spacing at 4 speeds × 2 spacings
# 8x spacing = wide gaps (beginner), 4x = moderate, matches Farnsworth style
TOP_500_WORDS = [
    f"Top 500 Words - {sx} Character Spacing - {w}wpm"
    for sx in ["8x", "4x"]
    for w in [15, 20, 25, 35]
]

# Top 100 Words — extra character-level variety at moderate speeds
TOP_100_WORDS = [
    f"Top 100 Words - {sx} Character Spacing - {w}wpm"
    for sx in ["6x", "3x"]
    for w in [20, 30]
]

ALL_PACKS = SINGLE_LETTERS + SINGLE_LETTER_NUMBER + TOP_500_WORDS + TOP_100_WORDS


def pack_url(name: str) -> str:
    encoded = urllib.parse.quote(name + ".zip")
    return CDN + encoded


def download(url: str, dest_path: str) -> bool:
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} — skipping")
        return False
    except Exception as e:
        print(f"  Error: {e} — skipping")
        return False


def extract(zip_path: str, out_dir: str):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)


def run(out_dir: str, dry_run: bool, keep_zips: bool):
    os.makedirs(out_dir, exist_ok=True)
    zip_dir = os.path.join(out_dir, '_zips')
    os.makedirs(zip_dir, exist_ok=True)

    print(f"{'DRY RUN — ' if dry_run else ''}Downloading {len(ALL_PACKS)} packs → '{out_dir}'\n")

    ok = skipped = 0

    for name in ALL_PACKS:
        url = pack_url(name)
        zip_path = os.path.join(zip_dir, name + '.zip')
        extract_dir = os.path.join(out_dir, name)

        print(f"  {name}")

        if dry_run:
            print(f"    {url}")
            continue

        if os.path.isdir(extract_dir):
            print(f"    already extracted, skipping")
            skipped += 1
            continue

        if not os.path.exists(zip_path):
            print(f"    downloading...", end=' ', flush=True)
            success = download(url, zip_path)
            if not success:
                skipped += 1
                continue
            print("done")
        else:
            print(f"    zip cached, extracting")

        print(f"    extracting...", end=' ', flush=True)
        os.makedirs(extract_dir, exist_ok=True)
        try:
            extract(zip_path, extract_dir)
            print("done")
            ok += 1
        except Exception as e:
            print(f"failed: {e}")
            skipped += 1

        if not keep_zips:
            os.remove(zip_path)

    if not dry_run:
        print(f"\nDownloaded: {ok}  Skipped/failed: {skipped}")
        print(f"\nNext step:")
        print(f"  python prepare_real_data.py {out_dir}/ --output-dir data/real")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Download Morse Code Ninja training packs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--out',       default='data/ninja_raw',
                    help='Directory for extracted audio (default: data/ninja_raw)')
    ap.add_argument('--dry-run',   action='store_true',
                    help='Print URLs without downloading')
    ap.add_argument('--keep-zips', action='store_true',
                    help='Keep ZIP files after extraction')
    args = ap.parse_args()

    run(args.out, args.dry_run, args.keep_zips)
