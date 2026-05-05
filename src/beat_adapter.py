"""
BEAT Dataset Adapter
=====================
Loads BEAT v1 facial blendshape data (52 ARKit weights at 60 FPS) and
converts it to the pipeline's 55-weight format at 30 FPS.

BEAT v1 stores facial data as JSON files with structure:
    {"names": ["browDownLeft", ...], "frames": [{"weights": [52 floats]}, ...]}

This adapter:
  1. Reads BEAT JSON files and remaps channels by name to pipeline order
  2. Pads 3 extra channels (headRoll, leftEyeRoll, rightEyeRoll) with zeros
  3. Downsamples from 60 FPS to 30 FPS (take every 2nd frame)
  4. Pairs with MFA phoneme alignments to build training datasets

BEAT v1 download (HuggingFace, Apache 2.0, no registration):
  pip install huggingface_hub
  huggingface-cli download H-Liu1997/BEAT --repo-type dataset --local-dir ./BEAT

  URL: https://huggingface.co/datasets/H-Liu1997/BEAT
  Version: beat_english_v0.2.1 (30 speakers, ~60h English)
  NOTE: Google Drive links in the BEAT GitHub README are dead (404).

Required directory structure after download:
  {beat_dir}/
    {speaker_id}/          # e.g., 1/, 2/, ... 30/
      *.json               # Facial blendshape JSON files (52 ARKit weights, 60 FPS)
      *.wav                # Audio WAV files (16kHz mono)
      *.TextGrid           # Word-level alignments (Praat TextGrid)
      *.bvh                # Body motion (not used by this adapter)
      *.csv                # Emotion labels (not used by this adapter)

Usage:
    python -m src.ml.beat_adapter /path/to/beat_rawdata_english --list
    python -m src.ml.beat_adapter /path/to/beat_rawdata_english --build-dataset
    python -m src.ml.beat_adapter /path/to/beat_rawdata_english --build-dataset --max-sequences 100
"""

import argparse
from datetime import time
import glob
import json
import os
import sys

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

NUM_WEIGHTS = 55  # target standard
BEAT_FPS = 60.0
TARGET_FPS = 30.0

# Target 55-channel order (from viseme_blendshape_map.json)
PIPELINE_ORDER = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft",
    "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight",
    "eyeLookOutRight", "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthLeft", "mouthRight", "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
    "headRoll", "leftEyeRoll", "rightEyeRoll",
]


def _build_channel_map(beat_names: list) -> list:
    """
    Build index mapping from BEAT channel order to pipeline order.

    Returns a list of length 55 where result[pipeline_idx] = beat_idx (or -1
    if the channel doesn't exist in BEAT, e.g., headRoll).
    """
    beat_lookup = {name: idx for idx, name in enumerate(beat_names)}
    mapping = []
    for pipeline_name in PIPELINE_ORDER:
        beat_idx = beat_lookup.get(pipeline_name, -1)
        mapping.append(beat_idx)
    return mapping


def _remap_weights(beat_weights: list, channel_map: list) -> list:
    """Remap a single frame's weights from BEAT order to pipeline order."""
    result = [0.0] * NUM_WEIGHTS
    for pipeline_idx, beat_idx in enumerate(channel_map):
        if beat_idx >= 0 and beat_idx < len(beat_weights):
            result[pipeline_idx] = beat_weights[beat_idx]
    return result


def load_beat_json(json_path: str) -> dict:
    """
    Load a BEAT v1 facial blendshape JSON file.

    Returns:
        {
            "frames": [{frame_index, time_ms, weights[55]}, ...],
            "fps": 30.0,
            "original_fps": 60.0,
            "n_original_frames": int,
            "beat_names": [52 channel names],
        }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    beat_names = data.get("names", [])
    beat_frames = data.get("frames", [])

    if not beat_names or not beat_frames:
        return {"frames": [], "fps": TARGET_FPS, "original_fps": BEAT_FPS,
                "n_original_frames": 0, "beat_names": beat_names}

    # Build channel mapping
    channel_map = _build_channel_map(beat_names)

    # Downsample from 60 FPS to 30 FPS (take every 2nd frame)
    downsample_step = max(1, round(BEAT_FPS / TARGET_FPS))
    ms_per_frame = 1000.0 / TARGET_FPS

    frames = []
    frame_idx = 0
    for i in range(0, len(beat_frames), downsample_step):
        beat_weights = beat_frames[i].get("weights", [])
        pipeline_weights = _remap_weights(beat_weights, channel_map)

        frames.append({
            "frame_index": frame_idx,
            "time_ms": round(frame_idx * ms_per_frame, 2),
            "weights": [round(w, 4) for w in pipeline_weights],
        })
        frame_idx += 1

    return {
        "frames": frames,
        "fps": TARGET_FPS,
        "original_fps": BEAT_FPS,
        "n_original_frames": len(beat_frames),
        "beat_names": beat_names,
    }


def discover_sequences(beat_dir: str) -> list:
    """
    Discover all BEAT sequences that have both JSON (facial) and WAV (audio).

    Supports two directory layouts:
      1. Per-speaker: {beat_dir}/{speaker_id}/*.json + *.wav (HuggingFace download)
      2. Per-modality: {beat_dir}/json/*.json + {beat_dir}/wav/*.wav (older layout)

    Returns list of dicts: [{id, json_path, wav_path, textgrid_path}, ...]
    """
    sequences = []

    # Try per-speaker layout first (HuggingFace BEAT v0.2.1)
    speaker_dirs = sorted(glob.glob(os.path.join(beat_dir, "*")))
    found_speaker_layout = False

    for speaker_path in speaker_dirs:
        if not os.path.isdir(speaker_path):
            continue

        json_files = glob.glob(os.path.join(speaker_path, "*.json"))
        if json_files:
            found_speaker_layout = True
            for json_path in sorted(json_files):
                seq_id = os.path.splitext(os.path.basename(json_path))[0]
                wav_path = os.path.join(speaker_path, seq_id + ".wav")
                tg_path = os.path.join(speaker_path, seq_id + ".TextGrid")

                if not os.path.isfile(wav_path):
                    continue

                sequences.append({
                    "id": seq_id,
                    "json_path": json_path,
                    "wav_path": wav_path,
                    "textgrid_path": tg_path if os.path.isfile(tg_path) else None,
                })

    if found_speaker_layout:
        return sequences

    # Fallback: per-modality layout
    json_dir = os.path.join(beat_dir, "json")
    wav_dir = os.path.join(beat_dir, "wav")
    tg_dir = os.path.join(beat_dir, "TextGrid")

    if not os.path.isdir(json_dir):
        print(f"  No BEAT data found in {beat_dir}")
        print(f"  Expected: per-speaker folders with *.json + *.wav")
        print(f"       or: json/ + wav/ subdirectories")
        return []

    for json_path in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        seq_id = os.path.splitext(os.path.basename(json_path))[0]
        wav_path = os.path.join(wav_dir, seq_id + ".wav")
        tg_path = os.path.join(tg_dir, seq_id + ".TextGrid")

        if not os.path.isfile(wav_path):
            continue

        sequences.append({
            "id": seq_id,
            "json_path": json_path,
            "wav_path": wav_path,
            "textgrid_path": tg_path if os.path.isfile(tg_path) else None,
        })

    return sequences


def extract_transcript_from_textgrid(tg_path: str) -> str:
    """
    Extract the full transcript from a BEAT TextGrid file.

    BEAT TextGrids have a "words" tier with word-level intervals.
    Only extracts from the words tier (not the phones tier).
    """
    if not tg_path or not os.path.isfile(tg_path):
        return ""

    try:
        with open(tg_path, "r", encoding="utf-8") as f:
            content = f.read()

        import re
        # Parse TextGrid to find the "words" tier and extract its intervals
        # We need to be careful to only get the words tier, not phones
        words = []
        in_words_tier = False
        for line in content.split("\n"):
            line = line.strip()
            if 'name = "words"' in line:
                in_words_tier = True
            elif 'name = "phones"' in line:
                in_words_tier = False
            elif in_words_tier and line.startswith('text = "'):
                text = line.split('"')[1].strip()
                if text:
                    words.append(text)

        return " ".join(words)
    except Exception:
        return ""


def extract_phonemes_from_textgrid(tg_path: str) -> list:
    """
    Extract phoneme timeline from a BEAT TextGrid file.

    BEAT TextGrids already contain a "phones" tier with ARPAbet phonemes
    and precise timing (captured via iPhone). No MFA needed.

    Returns:
        [{phone, viseme_10, start_s, duration_s}, ...]
    """
    if not tg_path or not os.path.isfile(tg_path):
        return []

    from src.arpabet import ARPABET_TO_VISEME

    try:
        with open(tg_path, "r", encoding="utf-8") as f:
            content = f.read()

        import re
        timeline = []
        in_phones_tier = False
        current_xmin = None
        current_xmax = None

        for line in content.split("\n"):
            line = line.strip()
            if 'name = "phones"' in line:
                in_phones_tier = True
            elif in_phones_tier and 'name = ' in line:
                break  # Left the phones tier

            if not in_phones_tier:
                continue

            if line.startswith("xmin = "):
                current_xmin = float(line.split("=")[1].strip())
            elif line.startswith("xmax = ") and current_xmin is not None:
                current_xmax = float(line.split("=")[1].strip())
            elif line.startswith('text = "') and current_xmin is not None:
                label = line.split('"')[1].strip()
                duration_s = current_xmax - current_xmin if current_xmax else 0.0

                if not label or duration_s <= 0:
                    current_xmin = None
                    current_xmax = None
                    continue

                # BEAT uses uppercase ARPAbet with stress digits: "IH0" → "ih"
                phone = label.rstrip("012").lower()
                if phone in ("", "sp", "spn", "sil"):
                    phone = "sil"

                viseme = ARPABET_TO_VISEME.get(phone, "sil")

                timeline.append({
                    "phone": phone,
                    "viseme_10": viseme,
                    "start_s": round(current_xmin, 4),
                    "duration_s": round(duration_s, 4),
                })

                current_xmin = None
                current_xmax = None

        return timeline
    except Exception:
        return []


def build_training_dataset(
    beat_dir: str,
    output_path: str = None,
    max_sequences: int = None,
    mfa_alignments_dir: str = None,
) -> str:
    """
    Build a TrainingDataset from BEAT data + MFA phoneme alignments.

    Args:
        beat_dir: Path to BEAT v1 raw data directory.
        output_path: Output dataset path (.json.gz).
        max_sequences: Limit number of sequences (for testing).
        mfa_alignments_dir: Directory with MFA TextGrid outputs (phoneme-level).
            If None, phoneme timeline will be empty (frames-only dataset).

    Returns:
        Path to saved dataset.
    """
    from src.data_format import TrainingDataset

    if output_path is None:
        ds_dir = os.path.join(_PROJECT_DIR, "data", "ml", "datasets")
        os.makedirs(ds_dir, exist_ok=True)
        output_path = os.path.join(ds_dir, "beat_v1.json.gz")

    sequences = discover_sequences(beat_dir)
    if max_sequences:
        sequences = sequences[:max_sequences]

    print(f"  Found {len(sequences)} sequences with audio")

    ds = TrainingDataset(provider="beat_v1", voice="human_iphone", fps=TARGET_FPS)

    loaded = 0
    skipped = 0
    t_start = time.time()

    for i, seq in enumerate(sequences):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(sequences) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(sequences)}] {rate:.0f} seq/s, ETA {eta:.0f}s")

        # Load blendshape frames
        beat_data = load_beat_json(seq["json_path"])
        frames = beat_data["frames"]

        if not frames:
            skipped += 1
            continue

        # Get transcript
        transcript = extract_transcript_from_textgrid(seq.get("textgrid_path"))

        # Get phoneme timeline: prefer BEAT's own phones tier (already has
        # ARPAbet phonemes with timing from iPhone capture). Fall back to
        # MFA alignments if BEAT TextGrid has no phones tier.
        phoneme_timeline = extract_phonemes_from_textgrid(seq.get("textgrid_path"))

        if not phoneme_timeline and mfa_alignments_dir:
            mfa_tg_path = os.path.join(mfa_alignments_dir, seq["id"] + ".TextGrid")
            phoneme_timeline = _parse_mfa_textgrid(mfa_tg_path)

        ds.add_sentence(
            sentence_id=seq["id"],
            text=transcript,
            phoneme_timeline=phoneme_timeline,
            target_frames=frames,
        )
        loaded += 1

    # Single save at the end (no checkpoints — they were re-serializing
    # the entire dataset each time, taking minutes on large builds)
    print(f"  Saving {loaded} sequences to {output_path}...")
    ds.save(output_path)
    print(f"  Loaded {loaded} sequences ({skipped} skipped)")
    print(f"  {ds.summary()}")
    print(f"  Saved: {output_path}")

    return output_path


def _parse_mfa_textgrid(tg_path: str) -> list:
    """
    Parse Montreal Forced Aligner TextGrid output to get phoneme timeline.

    MFA outputs TextGrid with a "phones" tier containing phoneme intervals
    with ARPAbet labels (e.g., "AH0", "T", "SIL").

    Returns:
        [{phone, viseme_10, start_s, duration_s}, ...]
    """
    if not tg_path or not os.path.isfile(tg_path):
        return []

    from src.arpabet import ARPABET_TO_VISEME

    try:
        import textgrid
        tg = textgrid.TextGrid.fromFile(tg_path)

        timeline = []
        for tier in tg:
            if tier.name.lower() in ("phones", "phone"):
                for interval in tier:
                    label = interval.mark.strip()
                    if not label or label in ("", "sp", "spn", "sil"):
                        phone = "sil"
                    else:
                        # MFA uses UPPERCASE ARPAbet with stress digits: "AH0" → "ah"
                        phone = label.rstrip("012").lower()

                    viseme = ARPABET_TO_VISEME.get(phone, "sil")
                    start_s = float(interval.minTime)
                    end_s = float(interval.maxTime)
                    duration_s = end_s - start_s

                    if duration_s > 0:
                        timeline.append({
                            "phone": phone,
                            "viseme_10": viseme,
                            "start_s": round(start_s, 4),
                            "duration_s": round(duration_s, 4),
                        })
                break

        return timeline

    except ImportError:
        # textgrid library not installed -- try regex fallback
        return _parse_mfa_textgrid_regex(tg_path)
    except Exception:
        return []


def _parse_mfa_textgrid_regex(tg_path: str) -> list:
    """Regex fallback parser for MFA TextGrid files."""
    import re
    from src.arpabet import ARPABET_TO_VISEME

    try:
        with open(tg_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return []

    # Find the phones tier and extract intervals
    # TextGrid format: intervals with xmin, xmax, text
    timeline = []
    # Simple pattern matching for intervals
    intervals = re.findall(
        r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"',
        content
    )

    for xmin, xmax, text in intervals:
        label = text.strip()
        if not label or label in ("", "sp", "spn", "sil"):
            phone = "sil"
        else:
            phone = label.rstrip("012").lower()

        viseme = ARPABET_TO_VISEME.get(phone, "sil")
        start_s = float(xmin)
        duration_s = float(xmax) - start_s

        if duration_s > 0:
            timeline.append({
                "phone": phone,
                "viseme_10": viseme,
                "start_s": round(start_s, 4),
                "duration_s": round(duration_s, 4),
            })

    return timeline


def main():
    parser = argparse.ArgumentParser(description="BEAT v1 dataset adapter")
    parser.add_argument("beat_dir", help="Path to BEAT v1 raw data directory")
    parser.add_argument("--list", action="store_true",
                        help="List available sequences and exit")
    parser.add_argument("--build-dataset", action="store_true",
                        help="Build training dataset from BEAT data")
    parser.add_argument("--output", default=None,
                        help="Output dataset path (.json.gz)")
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Limit number of sequences (for testing)")
    parser.add_argument("--mfa-dir", default=None,
                        help="Directory with MFA TextGrid alignments")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  BEAT v1 Dataset Adapter")
    print(f"{'=' * 60}")

    sequences = discover_sequences(args.beat_dir)
    print(f"  Directory: {args.beat_dir}")
    print(f"  Sequences found: {len(sequences)}")

    if not sequences:
        print("  No sequences found. Check directory structure.")
        print("  Expected: {beat_dir}/json/*.json + {beat_dir}/wav/*.wav")
        return

    # Extract speaker stats
    speakers = set()
    for seq in sequences:
        parts = seq["id"].split("_")
        if parts:
            speakers.add(parts[0])
    print(f"  Speakers: {len(speakers)}")

    if args.list:
        print(f"\n  First 20 sequences:")
        for seq in sequences[:20]:
            transcript = extract_transcript_from_textgrid(seq.get("textgrid_path"))
            preview = transcript[:60] + "..." if len(transcript) > 60 else transcript
            print(f"    {seq['id']}: {preview}")
        return

    if args.build_dataset:
        print(f"\n  Building training dataset...")
        build_training_dataset(
            beat_dir=args.beat_dir,
            output_path=args.output,
            max_sequences=args.max_sequences,
            mfa_alignments_dir=args.mfa_dir,
        )


if __name__ == "__main__":
    main()
