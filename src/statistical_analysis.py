"""
Statistical Analysis for Tier 1
=================================
Analyzes collected gold standard data to:
  1. Compute improved per-category weights (median, duration-weighted, clipped)
  2. Measure intra-category variance to identify categories that are too broad
  3. Run PCA per category to find sub-clusters
  4. Recommend category splits based on evidence

Usage:
    python -m src.ml.statistical_analysis path/to/dataset.json.gz
    python -m src.ml.statistical_analysis path/to/dataset.json.gz --update-map
    python -m src.ml.statistical_analysis path/to/dataset.json.gz --dry-run
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

VISEME_MAP_PATH = os.path.join(_PROJECT_DIR, "src", "viseme_blendshape_map.json")
REPORT_DIR = os.path.join(_PROJECT_DIR, "data", "reports")

# The 10 viseme categories in canonical order
CATEGORIES = ["sil", "aei", "o", "ee", "qw", "r", "l", "bmp", "fv", "th", "cdgknstxyz"]

NUM_WEIGHTS = 55

# Mouth channel indices for focused reporting
MOUTH_CHANNELS = {
    "jawOpen": 17, "mouthClose": 18, "mouthFunnel": 19, "mouthPucker": 20,
    "mouthSmileLeft": 23, "mouthSmileRight": 24,
    "mouthFrownLeft": 25, "mouthFrownRight": 26,
    "mouthStretchLeft": 29, "mouthStretchRight": 30,
    "mouthRollLower": 31, "mouthRollUpper": 32,
    "mouthShrugLower": 33, "mouthShrugUpper": 34,
    "mouthPressLeft": 35, "mouthPressRight": 36,
    "mouthLowerDownLeft": 37, "mouthLowerDownRight": 38,
    "mouthUpperUpLeft": 39, "mouthUpperUpRight": 40,
}


def analyze_dataset(dataset) -> dict:
    """
    Run full Tier 1 statistical analysis on collected training data.

    Args:
        dataset: TrainingDataset instance with gold standard data.

    Returns:
        Dict with keys: per_category_stats, improved_weights, pca_results,
        split_recommendations, transition_matrix, duration_stats.
    """
    from src.gold_standard import label_frames_by_viseme

    # Collect all labeled frames across all sentences
    all_labeled = []  # [(category, phone, weights), ...]
    all_durations = defaultdict(list)  # category -> [duration_s, ...]

    for sentence in dataset.sentences:
        frames = sentence.get("target_frames", [])
        viseme_events = sentence.get("viseme_events", [])
        timeline = sentence.get("phoneme_timeline", [])

        if frames and viseme_events:
            labeled = label_frames_by_viseme(frames, viseme_events)
            all_labeled.extend(labeled)

        # Collect durations per category from phoneme timeline
        for phoneme in timeline:
            cat = phoneme.get("viseme_10", "sil")
            dur = phoneme.get("duration_s", 0.0)
            all_durations[cat].append(dur)

    if not all_labeled:
        return {"error": "No labeled frames found in dataset"}

    # Group frames by category
    category_frames = defaultdict(list)
    category_phones = defaultdict(list)
    for category, phone, weights in all_labeled:
        category_frames[category].append(weights)
        category_phones[category].append(phone)

    # 1. Per-category statistics
    per_category_stats = _compute_per_category_stats(category_frames)

    # 2. Improved weights (median + duration-weighted + clipped)
    improved_weights = _compute_improved_weights(category_frames, all_durations)

    # 3. PCA per category
    pca_results = _run_pca_per_category(category_frames)

    # 4. Split recommendations
    split_recommendations = _recommend_splits(per_category_stats, pca_results, category_phones)

    # 5. Transition matrix
    transition_matrix = _compute_transition_matrix(dataset)

    # 6. Duration statistics
    duration_stats = _compute_duration_stats(all_durations)

    return {
        "per_category_stats": per_category_stats,
        "improved_weights": improved_weights,
        "pca_results": pca_results,
        "split_recommendations": split_recommendations,
        "transition_matrix": transition_matrix,
        "duration_stats": duration_stats,
        "frame_counts": {cat: len(frames) for cat, frames in category_frames.items()},
    }


def _compute_per_category_stats(category_frames: dict) -> dict:
    """Compute per-channel mean, median, stdev for each category."""
    stats = {}
    for cat in CATEGORIES:
        frames = category_frames.get(cat, [])
        if not frames:
            stats[cat] = {"count": 0}
            continue

        arr = np.array(frames)  # (N, 55)
        stats[cat] = {
            "count": len(frames),
            "mean": arr.mean(axis=0).tolist(),
            "median": np.median(arr, axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "p10": np.percentile(arr, 10, axis=0).tolist(),
            "p90": np.percentile(arr, 90, axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
        }
    return stats


def _compute_improved_weights(category_frames: dict, durations: dict) -> dict:
    """
    Compute improved viseme weights using multiple strategies.

    For each category, computes:
      - median: Robust to outlier transition frames
      - duration_weighted_mean: Frames from long phonemes count more
      - clipped_mean: Mean after clipping to 10th-90th percentile

    Returns the clipped_mean as the recommended weights (best balance of
    robustness and accuracy).
    """
    improved = {}
    for cat in CATEGORIES:
        frames = category_frames.get(cat, [])
        if not frames:
            improved[cat] = [0.0] * NUM_WEIGHTS
            continue

        arr = np.array(frames)

        # Method 1: Median
        median = np.median(arr, axis=0)

        # Method 2: Clipped mean (10th-90th percentile)
        p10 = np.percentile(arr, 10, axis=0)
        p90 = np.percentile(arr, 90, axis=0)
        clipped = np.clip(arr, p10, p90)
        clipped_mean = clipped.mean(axis=0)

        # Use clipped mean as primary (best balance)
        weights = np.clip(clipped_mean, 0.0, 1.0)
        improved[cat] = [round(float(w), 4) for w in weights]

    return improved


def _run_pca_per_category(category_frames: dict) -> dict:
    """
    Run PCA on each category's frames to detect sub-clusters.

    If the first principal component explains < 80% of variance, the category
    likely has distinct sub-poses that should be split.
    """
    results = {}
    for cat in CATEGORIES:
        frames = category_frames.get(cat, [])
        if len(frames) < 10:
            results[cat] = {"n_frames": len(frames), "skip": True}
            continue

        arr = np.array(frames)

        # Center the data
        mean = arr.mean(axis=0)
        centered = arr - mean

        # SVD for PCA (more numerically stable than covariance)
        try:
            _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            results[cat] = {"n_frames": len(frames), "skip": True, "error": "SVD failed"}
            continue

        # Explained variance ratios
        explained_var = singular_values ** 2
        total_var = explained_var.sum()
        if total_var < 1e-10:
            results[cat] = {"n_frames": len(frames), "explained_var_ratio": [1.0]}
            continue

        explained_ratio = (explained_var / total_var).tolist()

        # How many components for 95% variance?
        cumsum = np.cumsum(explained_var / total_var)
        n_for_95 = int(np.searchsorted(cumsum, 0.95) + 1)

        results[cat] = {
            "n_frames": len(frames),
            "explained_var_ratio_top5": [round(r, 4) for r in explained_ratio[:5]],
            "pc1_explains": round(explained_ratio[0], 4) if explained_ratio else 0.0,
            "n_components_for_95pct": n_for_95,
            "needs_split": explained_ratio[0] < 0.80 if explained_ratio else False,
        }

    return results


def _recommend_splits(stats: dict, pca: dict, category_phones: dict) -> list:
    """
    Recommend category splits based on statistical evidence.

    Criteria:
      - PCA first component explains < 80% of variance
      - High standard deviation on key mouth channels (jawOpen std > 0.08)
      - Multiple distinct phonemes within the category
    """
    recommendations = []

    for cat in CATEGORIES:
        cat_stats = stats.get(cat, {})
        cat_pca = pca.get(cat, {})
        phones = category_phones.get(cat, [])

        if cat_stats.get("count", 0) < 20:
            continue

        reasons = []

        # Check PCA
        if cat_pca.get("needs_split", False):
            pc1 = cat_pca.get("pc1_explains", 1.0)
            reasons.append(f"PC1 explains only {pc1:.0%} of variance (< 80%)")

        # Check key channel variance
        std = cat_stats.get("std", [])
        if std:
            jaw_std = std[17] if len(std) > 17 else 0.0
            if jaw_std > 0.08:
                reasons.append(f"jawOpen std={jaw_std:.3f} (high variance)")

        # Check phoneme diversity
        unique_phones = set(phones)
        if len(unique_phones) > 5:
            reasons.append(f"{len(unique_phones)} distinct phonemes in one category")

        if reasons:
            # Suggest concrete sub-groups for known broad categories
            suggestion = _suggest_subgroups(cat, unique_phones)
            recommendations.append({
                "category": cat,
                "reasons": reasons,
                "n_frames": cat_stats.get("count", 0),
                "unique_phones": sorted(unique_phones),
                "suggested_split": suggestion,
            })

    return recommendations


def _suggest_subgroups(category: str, phones: set) -> list:
    """Suggest articulatory sub-groups for a broad category."""
    if category == "cdgknstxyz":
        return [
            {"name": "alveolar_stops", "phones": ["d", "t", "n"]},
            {"name": "velar_stops", "phones": ["k", "g", "ng"]},
            {"name": "sibilants", "phones": ["s", "z", "sh", "zh", "ch", "jh"]},
            {"name": "glottal", "phones": ["hh", "y"]},
        ]
    elif category == "aei":
        return [
            {"name": "open_vowels", "phones": ["aa", "ae", "ah"]},
            {"name": "mid_vowels", "phones": ["eh", "ey"]},
            {"name": "diphthongs", "phones": ["ay", "aw"]},
        ]
    return []


def _compute_transition_matrix(dataset) -> dict:
    """Count pairwise viseme category transitions across all sentences."""
    transitions = defaultdict(int)

    for sentence in dataset.sentences:
        timeline = sentence.get("phoneme_timeline", [])
        for i in range(len(timeline) - 1):
            curr = timeline[i].get("viseme_10", "sil")
            nxt = timeline[i + 1].get("viseme_10", "sil")
            transitions[f"{curr}->{nxt}"] += 1

    return dict(transitions)


def _compute_duration_stats(durations: dict) -> dict:
    """Compute duration statistics per category."""
    stats = {}
    for cat in CATEGORIES:
        durs = durations.get(cat, [])
        if not durs:
            stats[cat] = {"count": 0}
            continue

        arr = np.array(durs) * 1000.0  # Convert to ms
        stats[cat] = {
            "count": len(durs),
            "mean_ms": round(float(arr.mean()), 1),
            "median_ms": round(float(np.median(arr)), 1),
            "std_ms": round(float(arr.std()), 1),
            "min_ms": round(float(arr.min()), 1),
            "max_ms": round(float(arr.max()), 1),
        }
    return stats


def generate_report(analysis: dict, output_path: str = None) -> str:
    """Generate a human-readable analysis report."""
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"Tier 1 Statistical Analysis Report -- {ts}")
    lines.append("=" * 70)

    # Frame counts
    counts = analysis.get("frame_counts", {})
    lines.append("\nFRAMES PER CATEGORY:")
    lines.append("-" * 50)
    for cat in CATEGORIES:
        n = counts.get(cat, 0)
        bar = "#" * min(n // 10, 40)
        lines.append(f"  {cat:<15} {n:>6} frames  {bar}")

    # Duration stats
    dur_stats = analysis.get("duration_stats", {})
    lines.append("\nDURATION STATISTICS (ms):")
    lines.append("-" * 60)
    lines.append(f"  {'Category':<15} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    for cat in CATEGORIES:
        ds = dur_stats.get(cat, {})
        if ds.get("count", 0) == 0:
            continue
        lines.append(
            f"  {cat:<15} {ds['mean_ms']:>8.1f} {ds['median_ms']:>8.1f} "
            f"{ds['std_ms']:>8.1f} {ds['min_ms']:>8.1f} {ds['max_ms']:>8.1f}"
        )

    # PCA results
    pca = analysis.get("pca_results", {})
    lines.append("\nPCA ANALYSIS (per category):")
    lines.append("-" * 60)
    lines.append(f"  {'Category':<15} {'Frames':>8} {'PC1 %':>8} {'Components for 95%':>20} {'Split?':>8}")
    for cat in CATEGORIES:
        p = pca.get(cat, {})
        if p.get("skip"):
            continue
        pc1 = p.get("pc1_explains", 0.0) * 100
        n95 = p.get("n_components_for_95pct", 0)
        split = "YES" if p.get("needs_split") else "no"
        lines.append(f"  {cat:<15} {p.get('n_frames', 0):>8} {pc1:>7.1f}% {n95:>20} {split:>8}")

    # Split recommendations
    recs = analysis.get("split_recommendations", [])
    if recs:
        lines.append("\nSPLIT RECOMMENDATIONS:")
        lines.append("=" * 70)
        for rec in recs:
            lines.append(f"\n  Category: {rec['category']} ({rec['n_frames']} frames)")
            for reason in rec["reasons"]:
                lines.append(f"    - {reason}")
            if rec.get("suggested_split"):
                lines.append("    Suggested sub-groups:")
                for sg in rec["suggested_split"]:
                    lines.append(f"      {sg['name']}: {', '.join(sg['phones'])}")
    else:
        lines.append("\nNo split recommendations (all categories appear homogeneous).")

    # Weight changes summary (mouth channels only)
    improved = analysis.get("improved_weights", {})
    if improved:
        lines.append("\nIMPROVED WEIGHTS (key mouth channels, delta from current map):")
        lines.append("-" * 70)

        try:
            with open(VISEME_MAP_PATH, "r", encoding="utf-8") as f:
                old_map = json.load(f)
        except FileNotFoundError:
            old_map = {}

        lines.append(f"  {'Category':<15} {'Channel':<25} {'Old':>6} {'New':>6} {'Delta':>8}")
        lines.append(f"  {'-' * 65}")

        for cat in CATEGORIES:
            if cat not in improved:
                continue
            old_weights = old_map.get(cat, [0.0] * NUM_WEIGHTS)
            new_weights = improved[cat]

            changes = []
            for name, idx in sorted(MOUTH_CHANNELS.items(), key=lambda x: x[1]):
                old_val = old_weights[idx] if idx < len(old_weights) else 0.0
                new_val = new_weights[idx]
                delta = new_val - old_val
                if abs(delta) > 0.005:
                    changes.append((name, old_val, new_val, delta))

            if changes:
                for name, old_val, new_val, delta in changes:
                    flag = " <<<" if abs(delta) > 0.1 else ""
                    lines.append(
                        f"  {cat:<15} {name:<25} {old_val:>6.3f} {new_val:>6.3f} {delta:>+8.4f}{flag}"
                    )

    report = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def update_viseme_map(improved_weights: dict, backup: bool = True):
    """
    Write improved weights to viseme_blendshape_map.json.

    Args:
        improved_weights: Dict of {category: [55 floats]}.
        backup: If True, save a backup before overwriting.
    """
    with open(VISEME_MAP_PATH, "r", encoding="utf-8") as f:
        original = json.load(f)

    if backup:
        backup_path = VISEME_MAP_PATH.replace(".json", "_backup.json")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(original, f, indent=2, ensure_ascii=False)
        print(f"  Backup saved: {backup_path}")

    for cat in CATEGORIES:
        if cat in improved_weights:
            original[cat] = improved_weights[cat]

    original["_comment"] = (
        f"Tier 1 optimized from gold standard data ({datetime.now().strftime('%Y-%m-%d')}). "
        f"Clipped-mean method (10th-90th percentile). "
        f"Maps 10 viseme categories to 55 ARKit blendshape weights."
    )

    with open(VISEME_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(original, f, indent=2, ensure_ascii=False)

    print(f"  Updated: {VISEME_MAP_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Tier 1 statistical analysis of training data")
    parser.add_argument("dataset", help="Path to training dataset (.json.gz or .json)")
    parser.add_argument("--update-map", action="store_true",
                        help="Write improved weights to viseme_blendshape_map.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze and report without writing changes")
    parser.add_argument("--report", default=None,
                        help="Path for analysis report (default: data/ml/reports/)")
    args = parser.parse_args()

    from src.data_format import TrainingDataset

    print(f"\n{'=' * 60}")
    print(f"  Tier 1 Statistical Analysis")
    print(f"{'=' * 60}")

    print(f"  Loading dataset: {args.dataset}")
    dataset = TrainingDataset.load(args.dataset)
    print(f"  {dataset.summary()}")

    print(f"\n  Analyzing...")
    analysis = analyze_dataset(dataset)

    if "error" in analysis:
        print(f"  ERROR: {analysis['error']}")
        return

    # Generate report
    report_path = args.report
    if report_path is None:
        os.makedirs(REPORT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(REPORT_DIR, f"tier1_analysis_{ts}.txt")

    report = generate_report(analysis, report_path)
    print(report)
    print(f"\n  Report saved: {report_path}")

    # Update map if requested
    if args.update_map and not args.dry_run:
        print(f"\n  Updating viseme map...")
        update_viseme_map(analysis["improved_weights"])
        print(f"\n  Run test_operator_comparison.py to verify improvement.")
    elif args.dry_run:
        print(f"\n  DRY RUN -- no changes written.")


if __name__ == "__main__":
    main()
