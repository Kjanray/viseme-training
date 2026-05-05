"""
Viseme Evaluation Metrics
==========================
Comprehensive evaluation metrics beyond simple MAE for comparing
predicted blendshape sequences against gold standard data.

Metrics:
  - MAE (mean absolute error, per channel)
  - RMSE (penalizes large errors)
  - Pearson correlation (temporal pattern match, scale-independent)
  - Velocity MAE (transition speed correctness)
  - Jerk (smoothness: 3rd derivative of weight curves)
  - DTW distance (alignment-robust similarity)
  - Perceptual-weighted MAE (visual salience weighting)
  - Key viseme hit rate (does the shape reach 80% of target peak?)
"""

import math
from collections import defaultdict

import numpy as np

NUM_WEIGHTS = 55

# Mouth channels that matter for lip sync evaluation
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

# Perceptual salience weights for each mouth channel (0-1 scale).
# Higher = more visually important for lip sync.
# Based on: jawOpen dominates perception, dimple/shrug barely visible.
PERCEPTUAL_WEIGHTS = {
    "jawOpen": 1.0,
    "mouthClose": 0.7,
    "mouthFunnel": 0.8,
    "mouthPucker": 0.8,
    "mouthSmileLeft": 0.5,
    "mouthSmileRight": 0.5,
    "mouthFrownLeft": 0.3,
    "mouthFrownRight": 0.3,
    "mouthStretchLeft": 0.6,
    "mouthStretchRight": 0.6,
    "mouthRollLower": 0.5,
    "mouthRollUpper": 0.4,
    "mouthShrugLower": 0.2,
    "mouthShrugUpper": 0.2,
    "mouthPressLeft": 0.6,
    "mouthPressRight": 0.6,
    "mouthLowerDownLeft": 0.7,
    "mouthLowerDownRight": 0.7,
    "mouthUpperUpLeft": 0.6,
    "mouthUpperUpRight": 0.6,
}

# Key channels that define each viseme category's recognizability
VISEME_KEY_CHANNELS = {
    "sil": [],  # Silence has no key activation
    "aei": ["jawOpen", "mouthStretchLeft", "mouthStretchRight"],
    "o": ["jawOpen", "mouthFunnel", "mouthPucker"],
    "ee": ["mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight"],
    "qw": ["mouthFunnel", "mouthPucker", "jawOpen"],
    "r": ["mouthFunnel", "mouthPucker"],
    "l": ["jawOpen"],  # tongueOut is idx 51, not in mouth channels
    "bmp": ["mouthPressLeft", "mouthPressRight"],
    "fv": ["mouthRollLower", "mouthUpperUpLeft", "mouthUpperUpRight"],
    "th": ["jawOpen"],  # tongueOut is key but not in mouth channels
    "cdgknstxyz": ["jawOpen", "mouthClose"],
}


def extract_weight_arrays(frames: list) -> np.ndarray:
    """Convert frame list to (N, 55) numpy array."""
    if not frames:
        return np.zeros((0, NUM_WEIGHTS))
    return np.array([f["weights"] for f in frames])


def time_align(gold_frames: np.ndarray, pred_frames: np.ndarray) -> tuple:
    """
    Align two frame sequences by time proportion for fair comparison.

    When sequences have different lengths (different speaking rates), resamples
    both to the same length using the longer sequence's length.

    Returns:
        (gold_aligned, pred_aligned) -- both shape (N, 55)
    """
    g_len = len(gold_frames)
    p_len = len(pred_frames)

    if g_len == 0 or p_len == 0:
        return np.zeros((0, NUM_WEIGHTS)), np.zeros((0, NUM_WEIGHTS))

    n = max(g_len, p_len)
    gold_aligned = np.zeros((n, NUM_WEIGHTS))
    pred_aligned = np.zeros((n, NUM_WEIGHTS))

    for i in range(n):
        g_idx = min(int(i * g_len / n), g_len - 1)
        p_idx = min(int(i * p_len / n), p_len - 1)
        gold_aligned[i] = gold_frames[g_idx]
        pred_aligned[i] = pred_frames[p_idx]

    return gold_aligned, pred_aligned


# ===================================================================
# Individual metrics
# ===================================================================

def mae_per_channel(gold: np.ndarray, pred: np.ndarray) -> dict:
    """Mean Absolute Error per mouth channel."""
    gold_a, pred_a = time_align(gold, pred)
    if len(gold_a) == 0:
        return {}

    result = {}
    for name, idx in MOUTH_CHANNELS.items():
        result[name] = float(np.abs(gold_a[:, idx] - pred_a[:, idx]).mean())
    return result


def rmse_per_channel(gold: np.ndarray, pred: np.ndarray) -> dict:
    """Root Mean Squared Error per mouth channel (penalizes large errors)."""
    gold_a, pred_a = time_align(gold, pred)
    if len(gold_a) == 0:
        return {}

    result = {}
    for name, idx in MOUTH_CHANNELS.items():
        mse = np.square(gold_a[:, idx] - pred_a[:, idx]).mean()
        result[name] = float(np.sqrt(mse))
    return result


def pearson_per_channel(gold: np.ndarray, pred: np.ndarray) -> dict:
    """
    Pearson correlation per mouth channel.

    Measures whether the temporal pattern is correct, independent of scale.
    A correlation of 0.95 with high MAE means "right shape, wrong magnitude"
    -- still perceptually acceptable.
    """
    gold_a, pred_a = time_align(gold, pred)
    if len(gold_a) < 3:
        return {}

    result = {}
    for name, idx in MOUTH_CHANNELS.items():
        g = gold_a[:, idx]
        p = pred_a[:, idx]

        # Handle constant channels (std = 0)
        if g.std() < 1e-6 or p.std() < 1e-6:
            result[name] = 1.0 if g.std() < 1e-6 and p.std() < 1e-6 else 0.0
            continue

        corr = np.corrcoef(g, p)[0, 1]
        result[name] = float(corr) if not np.isnan(corr) else 0.0

    return result


def velocity_mae_per_channel(gold: np.ndarray, pred: np.ndarray) -> dict:
    """
    MAE of frame-to-frame velocity (first derivative).

    Captures whether transitions have the correct speed, even if static
    poses are slightly off. Important because wrong transition speed
    makes lip sync look "floaty" or "jerky."
    """
    gold_a, pred_a = time_align(gold, pred)
    if len(gold_a) < 2:
        return {}

    # First derivative (velocity)
    gold_vel = np.diff(gold_a, axis=0)
    pred_vel = np.diff(pred_a, axis=0)

    result = {}
    for name, idx in MOUTH_CHANNELS.items():
        result[name] = float(np.abs(gold_vel[:, idx] - pred_vel[:, idx]).mean())
    return result


def jerk_metric(frames: np.ndarray) -> dict:
    """
    Average absolute jerk (3rd derivative) per mouth channel.

    Lower jerk = smoother, more natural movement. Gold standard's jerk
    sets the target; predicted should be similar (not necessarily lower).
    """
    if len(frames) < 4:
        return {}

    # Third derivative
    d3 = np.diff(frames, n=3, axis=0)

    result = {}
    for name, idx in MOUTH_CHANNELS.items():
        result[name] = float(np.abs(d3[:, idx]).mean())
    return result


def perceptual_weighted_mae(gold: np.ndarray, pred: np.ndarray) -> float:
    """
    Single scalar: perceptually-weighted MAE across all mouth channels.

    Weights each channel by visual importance (jawOpen matters most,
    mouthDimple barely visible). Returns 0.0 for perfect match.
    """
    mae = mae_per_channel(gold, pred)
    if not mae:
        return 0.0

    weighted_sum = 0.0
    weight_sum = 0.0
    for name, error in mae.items():
        w = PERCEPTUAL_WEIGHTS.get(name, 0.5)
        weighted_sum += error * w
        weight_sum += w

    return weighted_sum / weight_sum if weight_sum > 0 else 0.0


def key_viseme_hit_rate(
    gold: np.ndarray,
    pred: np.ndarray,
    phoneme_timeline: list,
    threshold: float = 0.80,
) -> dict:
    """
    For each viseme category, check if the predicted pose reaches the expected
    activation level on key channels.

    A "hit" means the predicted peak activation on a key channel reaches
    >= threshold * gold standard's peak for that channel during the same
    viseme's time window.

    Returns:
        {"overall_hit_rate": float, "per_category": {cat: {"hits": N, "total": N}}}
    """
    if len(gold) == 0 or len(pred) == 0 or not phoneme_timeline:
        return {"overall_hit_rate": 0.0, "per_category": {}}

    gold_a, pred_a = time_align(gold, pred)
    fps = 30.0
    ms_per_frame = 1000.0 / fps

    per_category = defaultdict(lambda: {"hits": 0, "total": 0})

    for phoneme in phoneme_timeline:
        cat = phoneme.get("viseme_10", "sil")
        key_channels = VISEME_KEY_CHANNELS.get(cat, [])
        if not key_channels:
            continue

        # Frame range for this phoneme
        start_ms = phoneme.get("start_s", 0.0) * 1000.0
        end_ms = start_ms + phoneme.get("duration_s", 0.0) * 1000.0
        start_frame = max(0, int(start_ms / ms_per_frame))
        end_frame = min(len(gold_a) - 1, int(end_ms / ms_per_frame))

        if start_frame >= end_frame:
            continue

        for ch_name in key_channels:
            idx = MOUTH_CHANNELS.get(ch_name)
            if idx is None:
                continue

            gold_peak = gold_a[start_frame:end_frame + 1, idx].max()
            pred_peak = pred_a[start_frame:end_frame + 1, idx].max()

            per_category[cat]["total"] += 1
            if gold_peak < 0.02:
                # Gold standard barely activates this channel -- auto-hit
                per_category[cat]["hits"] += 1
            elif pred_peak >= threshold * gold_peak:
                per_category[cat]["hits"] += 1

    # Compute rates
    total_hits = sum(v["hits"] for v in per_category.values())
    total_checks = sum(v["total"] for v in per_category.values())

    return {
        "overall_hit_rate": total_hits / total_checks if total_checks > 0 else 0.0,
        "per_category": dict(per_category),
    }


def dtw_distance(gold: np.ndarray, pred: np.ndarray, channel_idx: int = 17) -> float:
    """
    Dynamic Time Warping distance for a single channel.

    More robust than proportional alignment for handling timing differences.
    Uses jawOpen (idx 17) by default as the most perceptually important channel.

    Note: O(N*M) complexity. For full-channel DTW, call per-channel or use
    a subset of key channels.
    """
    if len(gold) == 0 or len(pred) == 0:
        return float("inf")

    g = gold[:, channel_idx]
    p = pred[:, channel_idx]
    n, m = len(g), len(p)

    # Full DTW matrix (optimize with windowing if needed for large sequences)
    dtw = np.full((n + 1, m + 1), float("inf"))
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(g[i - 1] - p[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m] / max(n, m))


# ===================================================================
# Aggregate evaluation
# ===================================================================

def evaluate_full(
    gold_frames: list,
    pred_frames: list,
    phoneme_timeline: list = None,
) -> dict:
    """
    Run all evaluation metrics on a pair of frame sequences.

    Args:
        gold_frames: Gold standard frames [{frame_index, time_ms, weights[55]}, ...]
        pred_frames: Predicted frames (same format)
        phoneme_timeline: Optional phoneme timeline for hit rate calculation

    Returns:
        Dict with all metric results.
    """
    gold = extract_weight_arrays(gold_frames)
    pred = extract_weight_arrays(pred_frames)

    results = {
        "n_gold_frames": len(gold),
        "n_pred_frames": len(pred),
        "mae": mae_per_channel(gold, pred),
        "rmse": rmse_per_channel(gold, pred),
        "pearson": pearson_per_channel(gold, pred),
        "velocity_mae": velocity_mae_per_channel(gold, pred),
        "perceptual_mae": perceptual_weighted_mae(gold, pred),
        "gold_jerk": jerk_metric(gold),
        "pred_jerk": jerk_metric(pred),
    }

    # DTW on key channels (expensive, so limit to top 3)
    dtw_channels = ["jawOpen", "mouthFunnel", "mouthPucker"]
    results["dtw"] = {}
    for ch_name in dtw_channels:
        idx = MOUTH_CHANNELS.get(ch_name)
        if idx is not None and len(gold) > 0 and len(pred) > 0:
            results["dtw"][ch_name] = dtw_distance(gold, pred, idx)

    # Viseme hit rate (if phoneme timeline available)
    if phoneme_timeline:
        results["viseme_hit_rate"] = key_viseme_hit_rate(
            gold, pred, phoneme_timeline
        )

    # Aggregate quality scores
    mae_vals = list(results["mae"].values())
    corr_vals = list(results["pearson"].values())

    results["avg_mouth_mae"] = sum(mae_vals) / len(mae_vals) if mae_vals else 0.0
    results["avg_mouth_corr"] = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
    results["quality_score"] = max(0.0, 1.0 - results["avg_mouth_mae"])

    return results


def format_evaluation_report(results: dict, label: str = "Evaluation") -> str:
    """Format evaluation results as a human-readable report."""
    lines = [f"\n  --- {label} ---"]
    lines.append(f"  Gold frames: {results['n_gold_frames']}, Pred frames: {results['n_pred_frames']}")

    # MAE table
    lines.append(f"\n  {'Channel':<25} {'MAE':>8} {'RMSE':>8} {'Corr':>8} {'Vel MAE':>8}")
    lines.append(f"  {'-' * 60}")

    for name in sorted(MOUTH_CHANNELS.keys(), key=lambda x: MOUTH_CHANNELS[x]):
        mae = results["mae"].get(name, 0.0)
        rmse = results["rmse"].get(name, 0.0)
        corr = results["pearson"].get(name, 0.0)
        vel = results["velocity_mae"].get(name, 0.0)
        lines.append(f"  {name:<25} {mae:>8.4f} {rmse:>8.4f} {corr:>8.3f} {vel:>8.4f}")

    # Aggregates
    lines.append(f"\n  Avg MAE:          {results['avg_mouth_mae']:.4f}")
    lines.append(f"  Avg Correlation:  {results['avg_mouth_corr']:.3f}")
    lines.append(f"  Perceptual MAE:   {results['perceptual_mae']:.4f}")
    lines.append(f"  Quality Score:    {results['quality_score']:.3f}")

    # DTW
    if results.get("dtw"):
        lines.append(f"\n  DTW distances:")
        for ch, dist in results["dtw"].items():
            lines.append(f"    {ch}: {dist:.4f}")

    # Hit rate
    if results.get("viseme_hit_rate"):
        hr = results["viseme_hit_rate"]
        lines.append(f"\n  Viseme Hit Rate:  {hr['overall_hit_rate']:.1%}")
        for cat, counts in sorted(hr.get("per_category", {}).items()):
            rate = counts["hits"] / counts["total"] if counts["total"] > 0 else 0.0
            lines.append(f"    {cat:<15} {rate:.1%} ({counts['hits']}/{counts['total']})")

    return "\n".join(lines)
