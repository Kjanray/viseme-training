"""
Training Data Collection
=========================
Synthesizes Harvard sentences through a gold standard provider and stores
the results in the standardized TrainingDataset format.

Supports resumption: if a partial dataset exists, skips already-collected
sentences. Estimates Azure TTS cost before running.

Usage:
    python -m src.ml.collect_data                         # Default: Azure, all 720
    python -m src.ml.collect_data --split train            # Train split only (570)
    python -m src.ml.collect_data --sentences 50           # First 50 sentences
    python -m src.ml.collect_data --provider azure --dry-run
    python -m src.ml.collect_data --resume path/to/partial.json.gz
"""

import argparse
import os
import sys
import time

# Ensure the project root is on sys.path so `src.*` imports work
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from src.harvard_sentences import (
    get_all_sentences,
    get_train_sentences,
    get_val_sentences,
    get_test_sentences,
)
from src.data_format import TrainingDataset
from src.gold_standard import AzureGoldStandard

# Default output directory
_DATA_DIR = os.path.join(_PROJECT_DIR, "data", "datasets")

# Cost estimates (per million characters)
_COST_RATES = {
    "azure": 15.0,  # $15 / 1M chars
}


def _estimate_cost(sentences: list[str], provider: str) -> float:
    """Estimate cost in USD for synthesizing all sentences."""
    total_chars = sum(len(s) for s in sentences)
    rate = _COST_RATES.get(provider, 15.0)
    return (total_chars / 1_000_000) * rate


def _make_sentence_id(index: int) -> str:
    """Generate a sentence ID like 'harvard_001_01' (list_sentence)."""
    list_num = (index // 10) + 1
    sentence_num = (index % 10) + 1
    return f"harvard_{list_num:03d}_{sentence_num:02d}"


def _get_provider(provider_name: str, output_dir: str):
    """Instantiate the gold standard provider."""
    if provider_name == "azure":
        audio_dir = os.path.join(output_dir, "audio")
        return AzureGoldStandard(output_dir=audio_dir)
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Supported: azure")


def collect(
    sentences: list[str],
    provider_name: str = "azure",
    output_path: str = None,
    resume_from: str = None,
    dry_run: bool = False,
):
    """
    Collect training data by synthesizing sentences through the gold standard.

    Args:
        sentences: List of sentence strings to synthesize.
        provider_name: Gold standard provider name ("azure").
        output_path: Path for the output dataset (.json.gz).
        resume_from: Path to a partial dataset to resume from.
        dry_run: If True, print cost estimate and exit.
    """
    # Cost estimate
    cost = _estimate_cost(sentences, provider_name)
    print(f"\n{'=' * 60}")
    print(f"  Training Data Collection")
    print(f"{'=' * 60}")
    print(f"  Provider:   {provider_name}")
    print(f"  Sentences:  {len(sentences)}")
    print(f"  Est. cost:  ${cost:.4f}")
    print(f"{'=' * 60}")

    if dry_run:
        print(f"\n  DRY RUN -- no synthesis performed.")
        print(f"  Characters: {sum(len(s) for s in sentences):,}")
        print(f"  Est. frames: ~{len(sentences) * 70:,} (avg 70 frames/sentence)")
        return

    # Set up output
    if output_path is None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        output_path = os.path.join(_DATA_DIR, f"harvard_{provider_name}.json.gz")

    # Resume from existing dataset if provided
    existing_ids = set()
    if resume_from and os.path.isfile(resume_from):
        print(f"\n  Resuming from: {resume_from}")
        dataset = TrainingDataset.load(resume_from)
        existing_ids = {s["id"] for s in dataset.sentences}
        print(f"  Already collected: {len(existing_ids)} sentences")
    else:
        dataset = TrainingDataset(provider=provider_name, voice="", fps=30.0)

    # Instantiate provider
    provider = _get_provider(provider_name, os.path.dirname(output_path))

    # Update voice in metadata from provider
    if hasattr(provider, "voice"):
        dataset.metadata["voice"] = provider.voice

    # Collect
    total = len(sentences)
    collected = 0
    failed = 0
    start_time = time.time()

    for i, text in enumerate(sentences):
        sid = _make_sentence_id(i)

        if sid in existing_ids:
            continue

        label = f'"{text[:45]}..."' if len(text) > 45 else f'"{text}"'
        print(f"  [{i+1}/{total}] {label}")

        try:
            result = provider.synthesize(text)
        except Exception as e:
            print(f"           FAILED: {e}")
            failed += 1
            continue

        frames = result.get("frames", [])
        if not frames:
            error = result.get("error", "no frames returned")
            print(f"           FAILED: {error}")
            failed += 1
            continue

        phoneme_timeline = result.get("phoneme_timeline", [])
        viseme_events = result.get("viseme_events", [])

        dataset.add_sentence(
            sentence_id=sid,
            text=text,
            phoneme_timeline=phoneme_timeline,
            target_frames=frames,
            viseme_events=viseme_events,
        )
        collected += 1
        print(f"           {len(frames)} frames, {len(phoneme_timeline)} phonemes")

        # Save checkpoint every 50 sentences
        if collected % 50 == 0:
            dataset.save(output_path)
            print(f"  [checkpoint] Saved {collected} sentences to {output_path}")

    # Final save
    elapsed = time.time() - start_time
    dataset.save(output_path)

    print(f"\n{'=' * 60}")
    print(f"  DONE")
    print(f"  Collected:  {collected} sentences ({failed} failed)")
    print(f"  Total:      {dataset.metadata['num_sentences']} sentences, "
          f"{dataset.metadata['total_frames']} frames")
    print(f"  Time:       {elapsed:.1f}s ({elapsed/max(collected,1):.1f}s per sentence)")
    print(f"  Output:     {output_path}")
    print(f"{'=' * 60}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Collect training data from gold standard TTS"
    )
    parser.add_argument(
        "--provider", default="azure", choices=["azure"],
        help="Gold standard provider (default: azure)"
    )
    parser.add_argument(
        "--split", default="all", choices=["all", "train", "val", "test"],
        help="Which sentence split to collect (default: all)"
    )
    parser.add_argument(
        "--sentences", type=int, default=None,
        help="Limit to first N sentences (overrides --split)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for dataset (.json.gz)"
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to partial dataset to resume from"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print cost estimate without synthesizing"
    )
    args = parser.parse_args()

    # Select sentences
    split_map = {
        "all": get_all_sentences,
        "train": get_train_sentences,
        "val": get_val_sentences,
        "test": get_test_sentences,
    }
    sentences = split_map[args.split]()

    if args.sentences is not None:
        sentences = sentences[: args.sentences]

    collect(
        sentences=sentences,
        provider_name=args.provider,
        output_path=args.output,
        resume_from=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
