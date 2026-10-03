"""
Compute _TARGET_MAX from BEAT training data.

Reads the built dataset and computes 95th percentile per blendshape channel
across all non-silence frames. Output is a Python dict ready to paste into
phoneme_inference.py.

Usage:
    python scripts/compute_target_max.py data/datasets/beat_v1.json.gz
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_format import TrainingDataset

MOUTH_CHANNELS = {
    17: "jawOpen", 18: "mouthClose", 19: "mouthFunnel", 20: "mouthPucker",
    23: "mouthSmileLeft", 24: "mouthSmileRight", 25: "mouthFrownLeft",
    26: "mouthFrownRight", 29: "mouthStretchLeft", 30: "mouthStretchRight",
    31: "mouthRollLower", 32: "mouthRollUpper", 33: "mouthShrugLower",
    34: "mouthShrugUpper", 35: "mouthPressLeft", 36: "mouthPressRight",
    37: "mouthLowerDownLeft", 38: "mouthLowerDownRight",
    39: "mouthUpperUpLeft", 40: "mouthUpperUpRight", 51: "tongueOut",
}


def main():
    parser = argparse.ArgumentParser(description="Compute _TARGET_MAX from BEAT data")
    parser.add_argument("dataset", help="Path to training dataset (.json.gz)")
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="Percentile to use as target max (default: 95)")
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    ds = TrainingDataset.load(args.dataset)

    all_weights = []
    for sentence in ds.sentences:
        for frame in sentence.get("target_frames", []):
            all_weights.append(frame["weights"])

    weights = np.array(all_weights, dtype=np.float32)
    print(f"  {weights.shape[0]} frames, {weights.shape[1]} channels")

    pct = np.percentile(weights, args.percentile, axis=0)

    print(f"\n_TARGET_MAX (p{args.percentile:.0f} from BEAT):")
    print("_TARGET_MAX = {")
    for idx, name in sorted(MOUTH_CHANNELS.items()):
        print(f"    {idx}: {pct[idx]:.2f},   # {name}")
    print("}")

    print(f"\nAll mouth channels p{args.percentile:.0f}:")
    print(f"  {'Channel':<25} {'p95':>6}  {'max':>6}  {'median':>6}")
    print(f"  {'-' * 50}")
    for idx, name in sorted(MOUTH_CHANNELS.items()):
        print(f"  {name:<25} {pct[idx]:>6.3f}  {weights[:, idx].max():>6.3f}  "
              f"{np.median(weights[:, idx]):>6.3f}")


if __name__ == "__main__":
    main()
