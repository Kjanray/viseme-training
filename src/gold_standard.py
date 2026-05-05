"""
Gold Standard Provider Interface
==================================
Abstraction layer for any TTS or mocap system that produces frame-level
55-weight blendshape data. Used during training data collection.

The provider interface is intentionally simple: text in, frames + optional
phoneme timeline out. This lets us swap Azure for mocap, another neural TTS,
or manual animator data without changing the training pipeline.
"""

import json
import os
import tempfile

# Azure's 22 viseme IDs mapped to our 10 viseme categories.
# Used to label frames for Tier 1/2 training.
# Source: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-speech-synthesis-viseme
AZURE_VISEME_TO_CATEGORY = {
    0: "sil",
    1: "aei",        # ae, ax, ah
    2: "aei",        # aa
    3: "o",          # ao
    4: "aei",        # ey, eh (dominant); uh is minor
    5: "r",          # er
    6: "ee",         # iy, ih (dominant); y is minor
    7: "qw",         # w, uw
    8: "o",          # ow
    9: "aei",        # aw (open diphthong)
    10: "o",         # oy
    11: "aei",       # ay
    12: "cdgknstxyz",  # h
    13: "r",         # r
    14: "l",         # l
    15: "cdgknstxyz",  # s, z
    16: "cdgknstxyz",  # sh, ch, jh, zh
    17: "th",        # th, dh
    18: "fv",        # f, v
    19: "cdgknstxyz",  # d, t, n
    20: "cdgknstxyz",  # k, g, ng
    21: "bmp",       # p, b, m
}

# Approximate phoneme for each Azure viseme ID (for Tier 3 training).
# When the gold standard only gives viseme IDs (not raw phonemes), this
# provides the best-guess phoneme. Grouped IDs use the most common phoneme.
AZURE_VISEME_TO_PHONE = {
    0: "sil",
    1: "ah",    # ae, ax, ah -- "ah" most common
    2: "aa",
    3: "ao",
    4: "ey",    # ey, eh, uh -- "ey" most common
    5: "er",
    6: "iy",    # y, iy, ih -- "iy" most common
    7: "w",     # w, uw
    8: "ow",
    9: "aw",
    10: "oy",
    11: "ay",
    12: "hh",
    13: "r",
    14: "l",
    15: "s",    # s, z
    16: "sh",   # sh, ch, jh, zh
    17: "th",   # th, dh
    18: "f",    # f, v
    19: "t",    # d, t, n
    20: "k",    # k, g, ng
    21: "p",    # p, b, m
}


class GoldStandardProvider:
    """
    Interface for any system that produces frame-level blendshape data.

    Subclasses must implement synthesize(). The returned dict must have
    at minimum "frames" (list of {frame_index, time_ms, weights[55]}).
    """

    def synthesize(self, text: str) -> dict:
        """
        Produce blendshape frames for the given text.

        Returns:
            {
                "frames": [{frame_index, time_ms, weights[55]}, ...],
                "viseme_events": [...],          # optional
                "phoneme_timeline": [...],       # optional
                "audio_file": str,               # optional path
            }
        """
        raise NotImplementedError


class AzureGoldStandard(GoldStandardProvider):
    """
    Gold standard using Azure Cognitive Services DragonHDOmni.

    Produces frame-level 55-weight blendshapes from Azure's neural model,
    plus viseme_id events that can be used to label frames by category.
    """

    def __init__(
        self,
        voice: str = "en-US-Ava:DragonHDOmniLatestNeural",
        output_dir: str = None,
        style: str = None,
    ):
        self.voice = voice
        self.style = style
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="azure_gold_")
        os.makedirs(self.output_dir, exist_ok=True)
        self._counter = 0

    def synthesize(self, text: str) -> dict:
        """
        Synthesize via Azure TTS and return frames + viseme events.

        The viseme_events include viseme_id fields that can be mapped to
        our 10 categories via AZURE_VISEME_TO_CATEGORY for frame labeling.
        """
        from src.tts.azure import text_to_speech_azure

        idx = self._counter
        self._counter += 1

        audio_path = os.path.join(self.output_dir, f"gold_{idx}.wav")
        json_path = os.path.join(self.output_dir, f"gold_{idx}.json")

        result = text_to_speech_azure(
            text=text,
            voice=self.voice,
            audio=audio_path,
            out=json_path,
            style=self.style,
        )

        if not result.get("success"):
            return {"frames": [], "error": result.get("message", "Azure TTS failed")}

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        frames = data.get("frames", [])
        viseme_events = data.get("viseme_events", [])

        # Build phoneme timeline from viseme events
        phoneme_timeline = self._build_phoneme_timeline(viseme_events)

        return {
            "frames": frames,
            "viseme_events": viseme_events,
            "phoneme_timeline": phoneme_timeline,
            "audio_file": audio_path,
        }

    def _build_phoneme_timeline(self, viseme_events: list) -> list:
        """
        Convert Azure viseme_id events into a phoneme timeline.

        Uses the non-animation viseme events (which carry viseme_id + timestamp)
        to build a timeline of approximate phonemes with durations.
        """
        # Filter to viseme_id markers (non-animation events)
        markers = []
        for evt in viseme_events:
            if evt.get("has_animation"):
                continue
            vid = evt.get("viseme_id", 0)
            time_ms = evt.get("audio_offset_ms", 0.0)
            markers.append((time_ms, vid))

        markers.sort(key=lambda x: x[0])

        if not markers:
            return []

        timeline = []
        for i, (time_ms, vid) in enumerate(markers):
            # Duration = time until next marker (or 50ms for last)
            if i < len(markers) - 1:
                duration_ms = markers[i + 1][0] - time_ms
            else:
                duration_ms = 50.0  # Default for final phoneme

            category = AZURE_VISEME_TO_CATEGORY.get(vid, "sil")
            phone = AZURE_VISEME_TO_PHONE.get(vid, "sil")

            timeline.append({
                "phone": phone,
                "viseme_10": category,
                "azure_viseme_id": vid,
                "start_s": time_ms / 1000.0,
                "duration_s": duration_ms / 1000.0,
            })

        return timeline


class FileGoldStandard(GoldStandardProvider):
    """
    Gold standard from pre-recorded files (e.g., mocap data).

    Reads blendshape frames from JSON files in a directory. Each file
    should match the standard output format (frames list with 55 weights).
    Optionally includes a phoneme timeline if alignment data is available.
    """

    def __init__(self, data_dir: str, fps: float = 30.0):
        self.data_dir = data_dir
        self.fps = fps
        self._file_map = {}  # text -> file path (built on first call)

    def set_file_map(self, text_to_file: dict):
        """Set explicit mapping from sentence text to JSON file path."""
        self._file_map = text_to_file

    def synthesize(self, text: str) -> dict:
        """Load pre-recorded frames from file."""
        file_path = self._file_map.get(text)
        if not file_path or not os.path.isfile(file_path):
            return {"frames": [], "error": f"No file found for: {text[:50]}..."}

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "frames": data.get("frames", []),
            "viseme_events": data.get("viseme_events", []),
            "phoneme_timeline": data.get("phoneme_timeline", []),
        }


class BeatGoldStandard(GoldStandardProvider):
    """
    Gold standard from the BEAT dataset (real iPhone ARKit capture).

    Loads pre-processed BEAT v1 facial JSON files (52 ARKit blendshapes)
    and optionally pairs them with MFA phoneme alignments.

    Unlike Azure which synthesizes on-demand, BEAT data is pre-recorded.
    The synthesize() method loads a pre-mapped sequence by text lookup.
    For bulk dataset building, use beat_adapter.build_training_dataset().
    """

    def __init__(self, beat_dir: str, mfa_alignments_dir: str = None):
        self.beat_dir = beat_dir
        self.mfa_alignments_dir = mfa_alignments_dir
        self._sequence_map = None  # Lazy-loaded

    def _ensure_sequence_map(self):
        """Build mapping from transcript text to sequence data."""
        if self._sequence_map is not None:
            return

        from src.beat_adapter import discover_sequences, extract_transcript_from_textgrid

        self._sequence_map = {}
        for seq in discover_sequences(self.beat_dir):
            transcript = extract_transcript_from_textgrid(seq.get("textgrid_path"))
            if transcript:
                self._sequence_map[transcript.lower().strip()] = seq

    def synthesize(self, text: str) -> dict:
        """Load BEAT frames for a matching transcript."""
        self._ensure_sequence_map()

        from src.beat_adapter import load_beat_json, _parse_mfa_textgrid

        seq = self._sequence_map.get(text.lower().strip())
        if not seq:
            return {"frames": [], "error": f"No BEAT sequence matches: {text[:50]}..."}

        beat_data = load_beat_json(seq["json_path"])
        frames = beat_data["frames"]

        phoneme_timeline = []
        if self.mfa_alignments_dir:
            mfa_path = os.path.join(self.mfa_alignments_dir, seq["id"] + ".TextGrid")
            phoneme_timeline = _parse_mfa_textgrid(mfa_path)

        return {
            "frames": frames,
            "phoneme_timeline": phoneme_timeline,
        }


def label_frames_by_viseme(frames: list, viseme_events: list) -> list:
    """
    Label each frame with its active viseme category using viseme_id events.

    This is the core function for Tier 1 training: it maps each gold-standard
    frame to one of the 10 viseme categories based on Azure's viseme_id markers.

    Args:
        frames: List of {frame_index, time_ms, weights[55]} dicts.
        viseme_events: List of viseme events with viseme_id and audio_offset_ms.

    Returns:
        List of (category, phone, weights) tuples.
    """
    # Build sorted markers from non-animation events
    markers = []
    for evt in viseme_events:
        if evt.get("has_animation"):
            continue
        vid = evt.get("viseme_id", 0)
        time_ms = evt.get("audio_offset_ms", 0.0)
        category = AZURE_VISEME_TO_CATEGORY.get(vid, "sil")
        phone = AZURE_VISEME_TO_PHONE.get(vid, "sil")
        markers.append((time_ms, category, phone))

    markers.sort(key=lambda x: x[0])

    if not markers:
        return []

    labeled = []
    for frame in frames:
        frame_time = frame.get("time_ms", 0.0)
        weights = frame.get("weights", [])

        if len(weights) != 55:
            continue

        # Find active viseme (last marker before frame time)
        category = "sil"
        phone = "sil"
        for marker_time, marker_cat, marker_phone in markers:
            if marker_time <= frame_time:
                category = marker_cat
                phone = marker_phone
            else:
                break

        labeled.append((category, phone, weights))

    return labeled
