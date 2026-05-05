"""
Montreal Forced Aligner (MFA) Wrapper
=======================================
Batch phoneme alignment for BEAT audio data using MFA.

MFA produces TextGrid files with precise ARPAbet phoneme timing for each
audio file, given the audio + transcript. These are then used by
beat_adapter.py to pair phoneme labels with blendshape frames.

Prerequisites:
    conda install -c conda-forge montreal-forced-aligner
    OR: pip install montreal-forced-aligner
    AND: mfa model download acoustic english_mfa
         mfa model download dictionary english_mfa

Usage:
    python -m src.ml.mfa_aligner /path/to/beat_rawdata_english --output /path/to/mfa_output
    python -m src.ml.mfa_aligner /path/to/beat_rawdata_english --check  # verify MFA installation
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)


def check_mfa_installation() -> bool:
    """Check if MFA is installed and models are available."""
    try:
        result = subprocess.run(
            ["mfa", "version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"  MFA version: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        pass

    print("  MFA not found. Install with:")
    print("    conda install -c conda-forge montreal-forced-aligner")
    print("    mfa model download acoustic english_mfa")
    print("    mfa model download dictionary english_mfa")
    return False


def prepare_mfa_corpus(beat_dir: str, corpus_dir: str, max_sequences: int = None):
    """
    Prepare a corpus directory for MFA from BEAT data.

    MFA expects: {corpus_dir}/{speaker_id}/{filename}.wav + .lab (or .txt)

    Creates symlinks to BEAT wav files and extracts transcripts from TextGrids
    into .lab text files.

    Args:
        beat_dir: BEAT v1 raw data directory.
        corpus_dir: Output corpus directory for MFA.
        max_sequences: Limit number of sequences.
    """
    from src.beat_adapter import discover_sequences, extract_transcript_from_textgrid

    sequences = discover_sequences(beat_dir)
    if max_sequences:
        sequences = sequences[:max_sequences]

    os.makedirs(corpus_dir, exist_ok=True)
    prepared = 0

    for seq in sequences:
        # Extract speaker ID from filename (e.g., "1_wayne_0_1_1" → speaker "1")
        parts = seq["id"].split("_")
        speaker_id = parts[0] if parts else "unknown"

        speaker_dir = os.path.join(corpus_dir, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)

        # Get transcript
        transcript = extract_transcript_from_textgrid(seq.get("textgrid_path"))
        if not transcript:
            continue

        # Copy/symlink WAV file
        wav_dest = os.path.join(speaker_dir, seq["id"] + ".wav")
        if not os.path.exists(wav_dest):
            try:
                os.symlink(os.path.abspath(seq["wav_path"]), wav_dest)
            except OSError:
                # Symlinks may fail on Windows -- copy instead
                shutil.copy2(seq["wav_path"], wav_dest)

        # Write transcript as .lab file
        lab_path = os.path.join(speaker_dir, seq["id"] + ".lab")
        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        prepared += 1

    print(f"  Prepared {prepared} files in {corpus_dir}")
    return prepared


def run_mfa_align(corpus_dir: str, output_dir: str, num_jobs: int = 4) -> bool:
    """
    Run MFA alignment on a prepared corpus.

    Args:
        corpus_dir: Directory with {speaker}/{file}.wav + .lab files.
        output_dir: Output directory for aligned TextGrid files.
        num_jobs: Number of parallel alignment jobs.

    Returns:
        True if alignment succeeded.
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "mfa", "align",
        corpus_dir,
        "english_mfa",          # Dictionary
        "english_mfa",          # Acoustic model
        output_dir,
        "--num_jobs", str(num_jobs),
        "--clean",              # Clean previous temp files
        "--overwrite",          # Overwrite existing outputs
    ]

    print(f"  Running: {' '.join(cmd)}")
    print(f"  This may take a while for large datasets...")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200  # 2 hour timeout
        )

        if result.returncode == 0:
            # Count output files
            n_outputs = len([f for f in os.listdir(output_dir)
                           if f.endswith(".TextGrid")])
            # MFA outputs into speaker subdirectories
            for speaker_dir in os.listdir(output_dir):
                speaker_path = os.path.join(output_dir, speaker_dir)
                if os.path.isdir(speaker_path):
                    for f in os.listdir(speaker_path):
                        if f.endswith(".TextGrid"):
                            # Move to flat output directory for easier loading
                            src = os.path.join(speaker_path, f)
                            dst = os.path.join(output_dir, f)
                            if not os.path.exists(dst):
                                shutil.move(src, dst)
                            n_outputs += 1

            print(f"  MFA alignment complete: {n_outputs} TextGrid files")
            return True
        else:
            print(f"  MFA alignment failed (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-10:]:
                    print(f"    {line}")
            return False

    except subprocess.TimeoutExpired:
        print("  MFA alignment timed out (2 hour limit)")
        return False
    except Exception as e:
        print(f"  MFA alignment error: {e}")
        return False


def align_beat_dataset(
    beat_dir: str,
    output_dir: str = None,
    max_sequences: int = None,
    num_jobs: int = 4,
) -> str:
    """
    Full pipeline: prepare BEAT data → run MFA → produce phoneme alignments.

    Args:
        beat_dir: BEAT v1 raw data directory.
        output_dir: Output directory for MFA TextGrid files.
        max_sequences: Limit number of sequences.
        num_jobs: Number of parallel MFA jobs.

    Returns:
        Path to output directory with aligned TextGrid files.
    """
    if output_dir is None:
        output_dir = os.path.join(_PROJECT_DIR, "data", "mfa_alignments")

    print(f"\n  Step 1: Preparing corpus...")
    corpus_dir = os.path.join(output_dir, "_corpus")
    n_prepared = prepare_mfa_corpus(beat_dir, corpus_dir, max_sequences)

    if n_prepared == 0:
        print("  No files prepared. Check BEAT directory structure.")
        return output_dir

    print(f"\n  Step 2: Running MFA alignment...")
    alignments_dir = os.path.join(output_dir, "alignments")
    success = run_mfa_align(corpus_dir, alignments_dir, num_jobs)

    if success:
        print(f"\n  Alignments saved to: {alignments_dir}")
        print(f"  Use with beat_adapter.py:")
        print(f"    python -m src.ml.beat_adapter {beat_dir} --build-dataset --mfa-dir {alignments_dir}")
    else:
        print(f"\n  MFA alignment failed. Check MFA installation:")
        print(f"    mfa model download acoustic english_mfa")
        print(f"    mfa model download dictionary english_mfa")

    return alignments_dir


def main():
    parser = argparse.ArgumentParser(description="MFA phoneme alignment for BEAT data")
    parser.add_argument("beat_dir", help="Path to BEAT v1 raw data directory")
    parser.add_argument("--output", default=None,
                        help="Output directory for MFA alignments")
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Limit number of sequences")
    parser.add_argument("--jobs", type=int, default=4,
                        help="Number of parallel MFA jobs (default: 4)")
    parser.add_argument("--check", action="store_true",
                        help="Check MFA installation and exit")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  MFA Phoneme Alignment for BEAT")
    print(f"{'=' * 60}")

    if args.check:
        check_mfa_installation()
        return

    if not check_mfa_installation():
        return

    align_beat_dataset(
        beat_dir=args.beat_dir,
        output_dir=args.output,
        max_sequences=args.max_sequences,
        num_jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
