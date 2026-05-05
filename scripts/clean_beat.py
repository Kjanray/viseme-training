"""
Clean BEAT dataset — delete files not needed for viseme training.

Keeps:  .json (facial blendshapes), .wav (audio), .TextGrid (transcripts)
Deletes: .bvh (body motion), .csv (emotion labels), .txt (semantic labels)

Usage:
    python scripts/clean_beat.py /path/to/BEAT                    # Preview (dry run)
    python scripts/clean_beat.py /path/to/BEAT --delete           # Actually delete
"""

import argparse
import glob
import os

KEEP_EXTENSIONS = {".json", ".wav", ".textgrid"}
DELETE_EXTENSIONS = {".bvh", ".csv", ".txt"}


def scan_beat_dir(beat_dir: str) -> dict:
    """Scan BEAT directory and categorize files by extension."""
    stats = {"keep": [], "delete": [], "other": []}
    total_delete_bytes = 0

    for root, dirs, files in os.walk(beat_dir):
        for f in files:
            path = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            size = os.path.getsize(path)

            if ext in KEEP_EXTENSIONS:
                stats["keep"].append((path, size))
            elif ext in DELETE_EXTENSIONS:
                stats["delete"].append((path, size))
                total_delete_bytes += size
            else:
                stats["other"].append((path, size))

    stats["total_delete_bytes"] = total_delete_bytes
    return stats


def format_size(nbytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Clean BEAT dataset (remove .bvh, .csv, .txt)")
    parser.add_argument("beat_dir", help="Path to BEAT dataset root directory")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete files (default: dry run preview)")
    args = parser.parse_args()

    print(f"\nScanning: {args.beat_dir}")
    stats = scan_beat_dir(args.beat_dir)

    # Summary by extension
    ext_sizes = {}
    for path, size in stats["delete"]:
        ext = os.path.splitext(path)[1].lower()
        ext_sizes.setdefault(ext, [0, 0])
        ext_sizes[ext][0] += 1
        ext_sizes[ext][1] += size

    keep_sizes = {}
    for path, size in stats["keep"]:
        ext = os.path.splitext(path)[1].lower()
        keep_sizes.setdefault(ext, [0, 0])
        keep_sizes[ext][0] += 1
        keep_sizes[ext][1] += size

    print(f"\n  KEEPING ({len(stats['keep'])} files):")
    for ext, (count, size) in sorted(keep_sizes.items()):
        print(f"    {ext:<12} {count:>5} files  {format_size(size):>10}")

    print(f"\n  DELETING ({len(stats['delete'])} files):")
    for ext, (count, size) in sorted(ext_sizes.items()):
        print(f"    {ext:<12} {count:>5} files  {format_size(size):>10}")

    print(f"\n  Space freed: {format_size(stats['total_delete_bytes'])}")

    if stats["other"]:
        print(f"\n  OTHER ({len(stats['other'])} files, untouched):")
        for path, size in stats["other"][:5]:
            print(f"    {os.path.basename(path)}")
        if len(stats["other"]) > 5:
            print(f"    ... and {len(stats['other']) - 5} more")

    if not args.delete:
        print(f"\n  DRY RUN — no files deleted.")
        print(f"  Re-run with --delete to actually remove files.")
        return

    # Delete
    deleted = 0
    for path, _ in stats["delete"]:
        try:
            os.remove(path)
            deleted += 1
        except OSError as e:
            print(f"  Failed to delete {path}: {e}")

    print(f"\n  Deleted {deleted}/{len(stats['delete'])} files.")
    print(f"  Freed {format_size(stats['total_delete_bytes'])}")


if __name__ == "__main__":
    main()
