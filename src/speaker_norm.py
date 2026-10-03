"""
Speaker Normalization
======================
Per-speaker mean subtraction so the model predicts deltas from each
speaker's neutral face instead of absolute blendshape weights.

Different speakers have different resting faces, jaw sizes, and lip
thickness. Without normalization, the model learns an average across
all speakers, which doesn't match any one speaker well.

Usage:
    means = compute_speaker_means(pairs)
    normalize_pairs(pairs, means)        # modifies target_weights in place
    denormalize_weights(weights, mean)   # restore absolute weights at inference
"""

from collections import defaultdict

import numpy as np

NUM_WEIGHTS = 55


def compute_speaker_means(pairs: list) -> dict:
    """
    Compute the mean blendshape vector per speaker.

    Returns:
        Dict mapping speaker_id -> np.ndarray of shape (55,).
    """
    accum = defaultdict(lambda: {"sum": np.zeros(NUM_WEIGHTS, dtype=np.float64), "n": 0})

    for p in pairs:
        sid = p.get("speaker_id", "")
        weights = p["target_weights"]
        accum[sid]["sum"] += np.array(weights[:NUM_WEIGHTS], dtype=np.float64)
        accum[sid]["n"] += 1

    return {
        sid: (data["sum"] / data["n"]).astype(np.float32)
        for sid, data in accum.items()
        if data["n"] > 0
    }


def normalize_pairs(pairs: list, speaker_means: dict):
    """
    Subtract each speaker's mean from target_weights (in place).

    After normalization, target_weights represent how far each phoneme
    deviates from that speaker's neutral face.
    """
    for p in pairs:
        sid = p.get("speaker_id", "")
        mean = speaker_means.get(sid)
        if mean is not None:
            raw = np.array(p["target_weights"][:NUM_WEIGHTS], dtype=np.float32)
            p["target_weights"] = (raw - mean).tolist()


def denormalize_weights(weights: np.ndarray, speaker_mean: np.ndarray) -> np.ndarray:
    """Add back the speaker mean to get absolute weights."""
    return np.clip(weights + speaker_mean, 0.0, 1.0)
