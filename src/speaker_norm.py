"""
Speaker Normalization
======================
Per-speaker mean subtraction so the model predicts deltas from each
speaker's neutral face instead of absolute blendshape weights.

Different speakers have different resting faces, jaw sizes, and lip
thickness. Without normalization, the model learns an average across
all speakers, which doesn't match any one speaker well.

At inference the speaker is unknown, so the trainer folds the average
neutral face (mean of the per-speaker means) back into the model bias.
The saved model outputs absolute weights and needs no extra parameters.

Usage:
    means = compute_speaker_means(train_pairs)       # train split only
    neutral = np.mean(list(means.values()), axis=0)
    Y_delta = Y - speaker_offsets(train_pairs, means, neutral)
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


def speaker_offsets(pairs: list, speaker_means: dict, fallback: np.ndarray) -> np.ndarray:
    """
    Per-pair speaker mean as an (N, 55) array. Speakers not in
    speaker_means (e.g. unseen in training) get the fallback vector.
    """
    return np.stack([
        speaker_means.get(p.get("speaker_id", ""), fallback) for p in pairs
    ]).astype(np.float32)
