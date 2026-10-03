"""
Training Dataset Format
========================
Provider-agnostic format for storing gold standard viseme training data.

The dataset stores (phoneme_timeline, target_frames) pairs per sentence.
Any provider that produces frame-level 55-weight blendshapes can populate
target_frames. The training pipeline never references provider-specific fields.

File format: gzip-compressed JSON (*.json.gz) to keep ~50K frames manageable.
"""

import gzip
import json
import os
from datetime import datetime


# Current schema version
SCHEMA_VERSION = "1.0"


class TrainingDataset:
    """
    Container for phoneme-to-blendshape training data.

    Attributes:
        metadata: Dict with provider info, creation date, sentence/frame counts.
        sentences: List of dicts, each with id, text, phoneme_timeline, target_frames.
    """

    def __init__(self, provider: str = "", voice: str = "", fps: float = 30.0):
        self.metadata = {
            "version": SCHEMA_VERSION,
            "provider": provider,
            "voice": voice,
            "fps": fps,
            "created": datetime.now().strftime("%Y-%m-%d"),
            "num_sentences": 0,
            "total_frames": 0,
        }
        self.sentences = []

    def add_sentence(
        self,
        sentence_id: str,
        text: str,
        phoneme_timeline: list,
        target_frames: list,
        viseme_events: list = None,
    ):
        """
        Add a training sentence with its phoneme alignment and gold-standard frames.

        Args:
            sentence_id: Unique ID (e.g., "harvard_001_01").
            text: The sentence text.
            phoneme_timeline: List of dicts:
                [{"phone": "dh", "viseme_10": "th", "start_s": 0.0, "duration_s": 0.05}, ...]
            target_frames: List of dicts (gold-standard blendshape frames):
                [{"frame_index": 0, "time_ms": 0.0, "weights": [55 floats]}, ...]
            viseme_events: Optional provider-specific viseme events (for provenance).
        """
        entry = {
            "id": sentence_id,
            "text": text,
            "phoneme_timeline": phoneme_timeline,
            "target_frames": target_frames,
        }
        if viseme_events:
            entry["viseme_events"] = viseme_events

        self.sentences.append(entry)
        self.metadata["num_sentences"] = len(self.sentences)
        self.metadata["total_frames"] += len(target_frames)

    def save(self, path: str):
        """
        Save dataset to gzip-compressed JSON.

        Args:
            path: Output path (should end in .json.gz).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        data = {
            "metadata": self.metadata,
            "sentences": self.sentences,
        }

        if path.endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))

    @classmethod
    def load(cls, path: str) -> "TrainingDataset":
        """
        Load dataset from JSON or gzip-compressed JSON.

        Args:
            path: Input path (.json or .json.gz).

        Returns:
            TrainingDataset instance with loaded data.
        """
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        ds = cls()
        ds.metadata = data.get("metadata", {})
        ds.sentences = data.get("sentences", [])
        return ds

    def get_all_labeled_frames(self, multi_sample: bool = False) -> list:
        """
        Extract all (phoneme_info, target_weights) pairs for training.

        Args:
            multi_sample: If True, extract frames at 25%, 50%, 75% of each
                phoneme's duration and average them for more robust targets.
                If False (default), extract only the midpoint frame.

        Returns:
            List of dicts with phone, context, duration, position,
            target_weights (55 floats), and sentence_idx.
        """
        all_pairs = []
        sample_points = [0.25, 0.5, 0.75] if multi_sample else [0.5]

        for sentence_idx, sentence in enumerate(self.sentences):
            timeline = sentence.get("phoneme_timeline", [])
            frames = sentence.get("target_frames", [])

            if not timeline or not frames:
                continue

            frame_times = [(f["time_ms"], f["weights"]) for f in frames]

            for i, phoneme in enumerate(timeline):
                phone = phoneme.get("phone", "sil")
                viseme = phoneme.get("viseme_10", "sil")
                start_s = phoneme.get("start_s", 0.0)
                duration_s = phoneme.get("duration_s", 0.0)

                sampled = []
                for frac in sample_points:
                    t_ms = (start_s + duration_s * frac) * 1000.0
                    w = _find_nearest_frame(frame_times, t_ms)
                    if w is not None:
                        sampled.append(w)

                if not sampled:
                    continue

                if len(sampled) == 1:
                    target = sampled[0]
                else:
                    target = [
                        sum(s[ch] for s in sampled) / len(sampled)
                        for ch in range(len(sampled[0]))
                    ]

                prev_phone = timeline[i - 1].get("phone", "sil") if i > 0 else "sil"
                next_phone = timeline[i + 1].get("phone", "sil") if i < len(timeline) - 1 else "sil"

                if i == 0:
                    position = "start"
                elif i == len(timeline) - 1:
                    position = "end"
                else:
                    position = "mid"

                # Speaker ID is the first segment of the sentence ID (e.g., "1" from "1_wayne_0_1_1")
                sentence_id = sentence.get("id", "")
                speaker_id = sentence_id.split("_")[0] if sentence_id else ""

                all_pairs.append({
                    "phone": phone,
                    "viseme_10": viseme,
                    "prev_phone": prev_phone,
                    "next_phone": next_phone,
                    "duration_s": duration_s,
                    "position": position,
                    "target_weights": target,
                    "sentence_idx": sentence_idx,
                    "speaker_id": speaker_id,
                })

        return all_pairs

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        lines = [
            f"TrainingDataset v{self.metadata.get('version', '?')}",
            f"  Provider:   {self.metadata.get('provider', '?')}",
            f"  Voice:      {self.metadata.get('voice', '?')}",
            f"  Created:    {self.metadata.get('created', '?')}",
            f"  Sentences:  {self.metadata.get('num_sentences', 0)}",
            f"  Frames:     {self.metadata.get('total_frames', 0)}",
            f"  FPS:        {self.metadata.get('fps', 30.0)}",
        ]
        return "\n".join(lines)


def _find_nearest_frame(frame_times: list, target_ms: float) -> list:
    """Find the frame weights closest to target_ms. Binary search."""
    if not frame_times:
        return None

    lo, hi = 0, len(frame_times) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if frame_times[mid][0] < target_ms:
            lo = mid + 1
        else:
            hi = mid

    # Check neighbors for closest
    best_idx = lo
    best_dist = abs(frame_times[lo][0] - target_ms)

    if lo > 0:
        dist = abs(frame_times[lo - 1][0] - target_ms)
        if dist < best_dist:
            best_idx = lo - 1

    return frame_times[best_idx][1]
