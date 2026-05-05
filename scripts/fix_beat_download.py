"""
Fix incomplete BEAT download — fetch only missing .json/.wav/.TextGrid files.

Compares what you have locally against the HuggingFace repo and downloads
only the needed files for speakers/sequences that are missing or incomplete.

Usage:
    python scripts/fix_beat_download.py                    # Preview what's missing
    python scripts/fix_beat_download.py --download         # Download missing files
    python scripts/fix_beat_download.py --download --speakers 8 9  # Specific speakers only
"""

import argparse
import os
import sys

BEAT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "BEAT", "beat_english_v0.2.1", "beat_english_v0.2.1"
)

REPO_ID = "H-Liu1997/BEAT"
KEEP_EXTENSIONS = (".json", ".wav", ".TextGrid")


def check_local(beat_dir: str) -> dict:
    """Check what's available locally per speaker."""
    local = {}
    if not os.path.isdir(beat_dir):
        return local

    for entry in sorted(os.listdir(beat_dir)):
        speaker_path = os.path.join(beat_dir, entry)
        if not os.path.isdir(speaker_path):
            continue
        try:
            speaker_id = int(entry)
        except ValueError:
            continue

        files = {}
        for f in os.listdir(speaker_path):
            base, ext = os.path.splitext(f)
            if ext in KEEP_EXTENSIONS:
                files.setdefault(base, set()).add(ext)

        local[speaker_id] = {
            "path": speaker_path,
            "sequences": files,
        }

    return local


def list_remote_files(speakers: list = None) -> dict:
    """List files on HuggingFace for given speakers (or all)."""
    from huggingface_hub import HfApi

    api = HfApi()
    remote = {}

    check_speakers = speakers or list(range(1, 31))

    for speaker_id in check_speakers:
        repo_path = f"beat_english_v0.2.1/beat_english_v0.2.1/{speaker_id}"
        try:
            items = list(api.list_repo_tree(
                repo_id=REPO_ID, repo_type="dataset", path_in_repo=repo_path
            ))
        except Exception:
            continue

        for item in items:
            path = getattr(item, "path", "")
            filename = path.rsplit("/", 1)[-1] if "/" in path else path
            if not filename or "." not in filename:
                continue

            base, ext = os.path.splitext(filename)
            if ext not in KEEP_EXTENSIONS:
                continue

            remote.setdefault(speaker_id, {}).setdefault(base, set()).add(ext)

    return remote


def find_missing(local: dict, remote: dict) -> list:
    """Find files that exist remotely but not locally."""
    missing = []

    for speaker_id in sorted(remote.keys()):
        local_seqs = local.get(speaker_id, {}).get("sequences", {})
        remote_seqs = remote[speaker_id]

        for seq_name, remote_exts in remote_seqs.items():
            local_exts = local_seqs.get(seq_name, set())
            for ext in remote_exts:
                if ext not in local_exts:
                    filename = seq_name + ext
                    missing.append({
                        "speaker": speaker_id,
                        "filename": filename,
                        "remote_path": f"beat_english_v0.2.1/beat_english_v0.2.1/{speaker_id}/{filename}",
                    })

    return missing


def download_missing(missing: list, beat_dir: str):
    """Download missing files from HuggingFace."""
    from huggingface_hub import hf_hub_download

    total = len(missing)
    downloaded = 0
    failed = 0

    for i, item in enumerate(missing):
        speaker_dir = os.path.join(beat_dir, str(item["speaker"]))
        os.makedirs(speaker_dir, exist_ok=True)

        dest = os.path.join(speaker_dir, item["filename"])
        if os.path.exists(dest):
            continue

        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=item["remote_path"],
                local_dir=os.path.join(beat_dir, "..", ".."),
            )
            downloaded += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{total}] Downloaded {downloaded} files...")
        except Exception as e:
            print(f"  FAILED: {item['filename']}: {e}")
            failed += 1

    print(f"\n  Downloaded: {downloaded}, Failed: {failed}, Skipped: {total - downloaded - failed}")


def main():
    parser = argparse.ArgumentParser(description="Fix incomplete BEAT download")
    parser.add_argument("--beat-dir", default=BEAT_DIR,
                        help="Path to BEAT data directory")
    parser.add_argument("--download", action="store_true",
                        help="Download missing files (default: preview only)")
    parser.add_argument("--speakers", type=int, nargs="*",
                        help="Only check/download specific speakers (e.g., --speakers 8 9)")
    args = parser.parse_args()

    beat_dir = os.path.abspath(args.beat_dir)
    print(f"\n{'=' * 60}")
    print(f"  BEAT Download Fixer")
    print(f"{'=' * 60}")
    print(f"  Local dir: {beat_dir}")

    # Check local state
    print(f"\n  Scanning local files...")
    local = check_local(beat_dir)
    local_speakers = sorted(local.keys())
    print(f"  Local speakers: {local_speakers}")
    print(f"  Expected speakers: 1-30")

    missing_speakers = [i for i in range(1, 31) if i not in local]
    if missing_speakers:
        print(f"  Missing speakers: {missing_speakers}")

    # Check which speakers to query
    target_speakers = args.speakers
    if target_speakers is None:
        # Default: check missing speakers + speakers with mismatches
        target_speakers = missing_speakers[:]
        # Also check all speakers for file mismatches
        for sid, info in local.items():
            seqs = info["sequences"]
            for base, exts in seqs.items():
                if len(exts) < 3:  # Should have .json, .wav, .TextGrid
                    if sid not in target_speakers:
                        target_speakers.append(sid)
                    break

    if not target_speakers:
        print(f"\n  All files appear complete. Nothing to download.")
        return

    target_speakers.sort()
    print(f"\n  Checking HuggingFace for speakers: {target_speakers}")
    print(f"  (fetching file list...)")

    remote = list_remote_files(target_speakers)
    print(f"  Remote speakers found: {sorted(remote.keys())}")

    # Find missing files
    missing = find_missing(local, remote)

    if not missing:
        print(f"\n  No missing files found. Download is complete.")
        return

    # Summarize by speaker
    by_speaker = {}
    for item in missing:
        by_speaker.setdefault(item["speaker"], []).append(item)

    print(f"\n  MISSING FILES ({len(missing)} total):")
    for speaker_id in sorted(by_speaker.keys()):
        files = by_speaker[speaker_id]
        exts = {}
        for f in files:
            ext = os.path.splitext(f["filename"])[1]
            exts[ext] = exts.get(ext, 0) + 1
        ext_str = ", ".join(f"{count}{ext}" for ext, count in sorted(exts.items()))
        print(f"    Speaker {speaker_id}: {len(files)} files ({ext_str})")

    if not args.download:
        print(f"\n  DRY RUN -- no files downloaded.")
        print(f"  Re-run with --download to fetch missing files.")
        return

    print(f"\n  Downloading {len(missing)} files...")
    download_missing(missing, beat_dir)
    print(f"  Done.")


if __name__ == "__main__":
    main()
