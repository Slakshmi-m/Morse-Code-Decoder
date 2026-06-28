"""
Merge a synthetic dataset (from generate_dataset.py) and a real dataset
(from prepare_real_data.py) into a single combined metadata file.

Files are referenced by absolute path — nothing is copied — so this is fast
and disk-efficient regardless of dataset size.

Usage:
  python mix_datasets.py                              # defaults
  python mix_datasets.py --synth data/synthetic --real data/real --out data/combined
  python mix_datasets.py --real-weight 3              # oversample real 3×

Then train on the combined set:
  python scripts/train.py --dataset data/combined
"""

import argparse
import json
import os
import random


def load_meta(dataset_dir: str) -> list:
    path = os.path.join(dataset_dir, 'metadata.json')
    with open(path) as f:
        return json.load(f)


def abs_file(dataset_dir: str, item: dict) -> str:
    """Return absolute path to the audio file referenced by a metadata entry."""
    f = item['file']
    if os.path.isabs(f):
        return f
    return os.path.abspath(os.path.join(dataset_dir, 'audio', f))


def mix(synth_dir: str, real_dir: str, out_dir: str, real_weight: float):
    os.makedirs(out_dir, exist_ok=True)

    synth_meta = load_meta(synth_dir)
    real_meta  = load_meta(real_dir)

    print(f"Synthetic samples : {len(synth_meta)}")
    print(f"Real samples      : {len(real_meta)}")

    combined = []

    # Synthetic entries — store absolute path so MorseDataset can find them
    for item in synth_meta:
        combined.append(dict(item, file=abs_file(synth_dir, item)))

    # Real entries — oversample if requested
    copies = max(1, round(real_weight))
    for item in real_meta:
        ap = abs_file(real_dir, item)
        for _ in range(copies):
            combined.append(dict(item, file=ap))

    random.shuffle(combined)

    meta_path = os.path.join(out_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(combined, f, indent=2)

    real_total = len(real_meta) * copies
    print(f"\nCombined : {len(combined)} samples  "
          f"({len(synth_meta)} synthetic + {real_total} real)")
    print(f"Saved to : '{out_dir}/metadata.json'")
    print(f"\nTrain:  python train.py --dataset {out_dir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Merge synthetic and real Morse datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--synth', default='data/synthetic',
                    help='Synthetic dataset dir (default: data/synthetic)')
    ap.add_argument('--real',  default='data/real',
                    help='Real dataset dir (default: data/real)')
    ap.add_argument('--out',   default='data/combined',
                    help='Output dir for combined metadata (default: data/combined)')
    ap.add_argument('--real-weight', type=float, default=1.0,
                    help='Oversample real data N× (e.g. 3 = three copies). Default: 1.0')
    args = ap.parse_args()

    mix(args.synth, args.real, args.out, args.real_weight)
