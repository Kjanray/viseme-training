"""
CMUdict Phoneme Disambiguator
================================
Resolves Azure's ambiguous viseme IDs (e.g., ID 19 = "d"/"t"/"n") into
specific ARPAbet phonemes using the CMU Pronouncing Dictionary.

Azure maps ~40 phonemes into 22 viseme IDs. This module recovers the
original ~40 phonemes by looking up each word's pronunciation in CMUdict
and aligning the resulting phoneme sequence with the Azure viseme timeline.

Usage:
    python -m src.ml.cmudict_disambiguator data/ml/datasets/harvard_azure.json.gz
    python -m src.ml.cmudict_disambiguator data/ml/datasets/harvard_azure.json.gz --output data/ml/datasets/harvard_azure_disambig.json.gz
"""

import argparse
import os
import re
import sys

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# Which Azure viseme IDs are ambiguous (map to multiple phonemes)
# Format: {viseme_id: set of possible ARPAbet phonemes}
AMBIGUOUS_IDS = {
    1: {"ae", "ax", "ah"},
    4: {"ey", "eh", "uh"},
    6: {"y", "iy", "ih"},
    7: {"w", "uw"},
    9: {"aw"},
    10: {"oy"},
    11: {"ay"},
    15: {"s", "z"},
    16: {"sh", "ch", "jh", "zh"},
    17: {"th", "dh"},
    18: {"f", "v"},
    19: {"d", "t", "n"},
    20: {"k", "g", "ng"},
    21: {"p", "b", "m"},
}

# Full mapping: Azure viseme ID → set of all phonemes it could represent
VISEME_ID_TO_PHONES = {
    0: {"sil"},
    1: {"ae", "ax", "ah"},
    2: {"aa"},
    3: {"ao"},
    4: {"ey", "eh", "uh"},
    5: {"er"},
    6: {"y", "iy", "ih"},
    7: {"w", "uw"},
    8: {"ow"},
    9: {"aw"},
    10: {"oy"},
    11: {"ay"},
    12: {"hh"},
    13: {"r"},
    14: {"l"},
    15: {"s", "z"},
    16: {"sh", "ch", "jh", "zh"},
    17: {"th", "dh"},
    18: {"f", "v"},
    19: {"d", "t", "n"},
    20: {"k", "g", "ng"},
    21: {"p", "b", "m"},
}

# Cached CMUdict
_cmudict = None


def _load_cmudict() -> dict:
    """Load CMU Pronouncing Dictionary. Returns {word: [phonemes]}."""
    global _cmudict
    if _cmudict is not None:
        return _cmudict

    try:
        import nltk
        from nltk.corpus import cmudict
        try:
            entries = cmudict.entries()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            entries = cmudict.entries()

        _cmudict = {}
        for word, phones in entries:
            if word not in _cmudict:
                # Strip stress digits: "AH0" → "ah"
                _cmudict[word] = [p.rstrip("012").lower() for p in phones]
        return _cmudict

    except ImportError:
        print("  WARNING: nltk not installed. Install with: pip install nltk")
        _cmudict = {}
        return _cmudict


def _tokenize(text: str) -> list:
    """Extract words from text, lowercase, strip punctuation."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def _get_sentence_phonemes(text: str) -> list:
    """
    Get the expected phoneme sequence for a sentence using CMUdict.

    Returns a flat list of ARPAbet phonemes (lowercase, no stress digits).
    Words not in CMUdict are skipped.
    """
    cmudict = _load_cmudict()
    words = _tokenize(text)
    phonemes = []

    for word in words:
        # Handle contractions
        clean = word.replace("'", "")
        lookup = cmudict.get(word) or cmudict.get(clean)
        if lookup:
            phonemes.extend(lookup)

    return phonemes


def disambiguate_timeline(text: str, azure_timeline: list) -> list:
    """
    Resolve ambiguous Azure viseme IDs to specific ARPAbet phonemes.

    Uses greedy alignment between CMUdict's phoneme sequence and Azure's
    viseme event sequence. For each Azure event, checks if the next
    CMUdict phoneme could have produced this viseme ID.

    Args:
        text: The sentence text (used for CMUdict lookup).
        azure_timeline: List of dicts with "azure_viseme_id" and "phone" keys.

    Returns:
        New timeline with "phone" fields replaced by specific phonemes
        where disambiguation was possible. Original "phone" is preserved
        as "phone_original".
    """
    expected_phones = _get_sentence_phonemes(text)

    if not expected_phones:
        # No CMUdict data — return original
        return azure_timeline

    # Filter out silence events for alignment (silence doesn't correspond to CMUdict phonemes)
    speech_events = [(i, evt) for i, evt in enumerate(azure_timeline)
                     if evt.get("azure_viseme_id", 0) != 0]

    # Greedy alignment: walk through both sequences
    cmu_idx = 0
    disambiguated = [dict(evt) for evt in azure_timeline]  # Deep copy

    for orig_idx, evt in speech_events:
        vid = evt.get("azure_viseme_id", 0)
        possible_phones = VISEME_ID_TO_PHONES.get(vid, set())

        if cmu_idx >= len(expected_phones):
            break

        cmu_phone = expected_phones[cmu_idx]

        if cmu_phone in possible_phones:
            # Match! Use the specific CMUdict phoneme
            disambiguated[orig_idx]["phone_original"] = disambiguated[orig_idx]["phone"]
            disambiguated[orig_idx]["phone"] = cmu_phone
            cmu_idx += 1
        else:
            # Mismatch — try skipping CMUdict phonemes that might have been
            # absorbed into the previous Azure event (coarticulation)
            matched = False
            for lookahead in range(1, 4):  # Look ahead up to 3 positions
                if cmu_idx + lookahead >= len(expected_phones):
                    break
                if expected_phones[cmu_idx + lookahead] in possible_phones:
                    # Found match after skipping
                    cmu_idx += lookahead
                    disambiguated[orig_idx]["phone_original"] = disambiguated[orig_idx]["phone"]
                    disambiguated[orig_idx]["phone"] = expected_phones[cmu_idx]
                    cmu_idx += 1
                    matched = True
                    break

            if not matched:
                # No match found — keep original label, advance CMUdict
                cmu_idx += 1

    return disambiguated


def disambiguate_dataset(dataset) -> "TrainingDataset":
    """
    Create a new dataset with disambiguated phoneme labels.

    For each sentence in the dataset, resolves ambiguous Azure viseme IDs
    to specific ARPAbet phonemes using CMUdict.

    Returns a new TrainingDataset with updated phoneme timelines.
    """
    from src.data_format import TrainingDataset

    new_ds = TrainingDataset(
        provider=dataset.metadata.get("provider", "") + "+cmudict",
        voice=dataset.metadata.get("voice", ""),
        fps=dataset.metadata.get("fps", 30.0),
    )

    total = len(dataset.sentences)
    disambiguated_count = 0
    total_events = 0
    changed_events = 0

    for i, sentence in enumerate(dataset.sentences):
        text = sentence.get("text", "")
        timeline = sentence.get("phoneme_timeline", [])

        # Disambiguate
        new_timeline = disambiguate_timeline(text, timeline)

        # Count changes
        for j, (old, new) in enumerate(zip(timeline, new_timeline)):
            total_events += 1
            if new.get("phone") != old.get("phone"):
                changed_events += 1

        if any(new.get("phone") != old.get("phone")
               for old, new in zip(timeline, new_timeline)):
            disambiguated_count += 1

        new_ds.add_sentence(
            sentence_id=sentence.get("id", f"s_{i}"),
            text=text,
            phoneme_timeline=new_timeline,
            target_frames=sentence.get("target_frames", []),
            viseme_events=sentence.get("viseme_events", []),
        )

    print(f"  Disambiguated {disambiguated_count}/{total} sentences")
    print(f"  Changed {changed_events}/{total_events} phoneme labels "
          f"({100*changed_events/max(total_events,1):.1f}%)")

    return new_ds


def main():
    parser = argparse.ArgumentParser(
        description="Disambiguate Azure phoneme labels using CMUdict"
    )
    parser.add_argument("dataset", help="Path to training dataset (.json.gz)")
    parser.add_argument("--output", default=None,
                        help="Output path (default: append '_disambig' to input name)")
    args = parser.parse_args()

    from src.data_format import TrainingDataset

    print(f"\n{'=' * 60}")
    print(f"  CMUdict Phoneme Disambiguation")
    print(f"{'=' * 60}")

    print(f"  Loading: {args.dataset}")
    ds = TrainingDataset.load(args.dataset)
    print(f"  {ds.metadata.get('num_sentences', 0)} sentences")

    # Check CMUdict is available
    cmudict = _load_cmudict()
    print(f"  CMUdict entries: {len(cmudict)}")
    if not cmudict:
        print("  ERROR: CMUdict not available. Install nltk: pip install nltk")
        return

    print(f"\n  Disambiguating...")
    new_ds = disambiguate_dataset(ds)

    # Show vocabulary comparison
    old_pairs = ds.get_all_labeled_frames()
    new_pairs = new_ds.get_all_labeled_frames()

    old_vocab = sorted(set(p["phone"] for p in old_pairs))
    new_vocab = sorted(set(p["phone"] for p in new_pairs))

    print(f"\n  Old vocabulary ({len(old_vocab)}): {old_vocab}")
    print(f"  New vocabulary ({len(new_vocab)}): {new_vocab}")
    print(f"  New phonemes: {sorted(set(new_vocab) - set(old_vocab))}")

    # Save
    output = args.output
    if output is None:
        base, ext = os.path.splitext(args.dataset)
        if base.endswith(".json"):
            base = base[:-5]
            ext = ".json" + ext
        output = base + "_disambig" + ext

    new_ds.save(output)
    print(f"\n  Saved: {output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
