"""
Phoneme Model Version Manager
================================
Save, list, activate, and compare different trained model versions.

The active model is always at data/ml/models/phoneme_ridge.npz — this is
what viseme_mapper.py loads at runtime. Named versions are stored alongside
it and can be activated (copied to the active path) with one command.

Usage:
    python -m src.ml.model_manager list                              # List all saved versions
    python -m src.ml.model_manager save azure39 "CMUdict disambig"   # Save current as named version
    python -m src.ml.model_manager activate azure39                  # Switch active model
    python -m src.ml.model_manager compare azure22 azure39 beat      # Compare versions side by side
    python -m src.ml.model_manager info azure39                      # Show model metadata
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

MODEL_DIR = os.path.join(_PROJECT_DIR, "data", "models")
ACTIVE_MODEL = os.path.join(MODEL_DIR, "phoneme_ridge.npz")


def _version_path(name: str) -> str:
    """Get path for a named model version."""
    return os.path.join(MODEL_DIR, f"phoneme_ridge_{name}.npz")


def list_versions():
    """List all saved model versions."""
    print(f"\n  Model directory: {MODEL_DIR}")
    print(f"  Active model:    {ACTIVE_MODEL}")
    print(f"  {'':->60}")

    if not os.path.isdir(MODEL_DIR):
        print(f"  No models directory found.")
        return

    # Find all model files
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "phoneme_ridge*.npz")))

    if not files:
        print(f"  No models found.")
        return

    # Check which one is active (by content comparison)
    active_hash = None
    if os.path.isfile(ACTIVE_MODEL):
        active_hash = _file_hash(ACTIVE_MODEL)

    print(f"\n  {'Name':<20} {'Type':>6} {'Phonemes':>8} {'Params':>8} {'Val MAE':>10} {'Quality':>8} {'Active':>8}")
    print(f"  {'-' * 72}")

    for path in files:
        filename = os.path.basename(path)
        if filename == "phoneme_ridge.npz":
            name = "(active)"
        else:
            name = filename.replace("phoneme_ridge_", "").replace(".npz", "")

        try:
            data = np.load(path, allow_pickle=False)
            vocab = data["vocab"].tolist()
            n_phonemes = len(vocab)

            model_type = str(data["model_type"]) if "model_type" in data else "ridge"

            if model_type == "mlp":
                n_params = sum(
                    data[f"W{i}"].size + data[f"b{i}"].size
                    for i in range(1, 4)
                )
            else:
                W = data["W"]
                n_params = W.shape[0] * W.shape[1] + W.shape[0]

            meta = {}
            if "metadata_json" in data:
                meta = json.loads(str(data["metadata_json"]))

            val_mae = meta.get("val_avg_mouth_mae", "?")
            quality = meta.get("val_quality_score", "?")

            if isinstance(val_mae, float):
                val_mae = f"{val_mae:.4f}"
            if isinstance(quality, float):
                quality = f"{quality:.3f}"

            is_active = ""
            if active_hash and _file_hash(path) == active_hash and name != "(active)":
                is_active = "<--"

            print(f"  {name:<20} {model_type:>6} {n_phonemes:>8} {n_params:>8,} {val_mae:>10} {quality:>8} {is_active:>8}")

        except Exception as e:
            print(f"  {name:<20} ERROR: {e}")


def save_version(name: str, description: str = ""):
    """Save the current active model as a named version."""
    if not os.path.isfile(ACTIVE_MODEL):
        print(f"  No active model to save. Train one first.")
        return

    dest = _version_path(name)
    if os.path.exists(dest):
        print(f"  Version '{name}' already exists at {dest}")
        print(f"  Delete it first or choose a different name.")
        return

    shutil.copy2(ACTIVE_MODEL, dest)
    print(f"  Saved: {name} -> {dest}")

    if description:
        # Append description to metadata
        try:
            data = dict(np.load(dest, allow_pickle=False))
            meta = {}
            if "metadata_json" in data:
                meta = json.loads(str(data["metadata_json"]))
            meta["description"] = description
            data["metadata_json"] = np.array(json.dumps(meta), dtype="U")
            np.savez_compressed(dest, **data)
            print(f"  Description: {description}")
        except Exception:
            pass


def activate_version(name: str):
    """Copy a named version to the active model path."""
    src = _version_path(name)
    if not os.path.isfile(src):
        print(f"  Version '{name}' not found at {src}")
        print(f"  Available versions:")
        list_versions()
        return

    # Backup current active model if it exists and isn't already saved
    if os.path.isfile(ACTIVE_MODEL):
        backup = os.path.join(MODEL_DIR, "phoneme_ridge_prev.npz")
        shutil.copy2(ACTIVE_MODEL, backup)

    shutil.copy2(src, ACTIVE_MODEL)
    print(f"  Activated: {name}")
    print(f"  viseme_mapper.py will use this model on next request.")

    # Reset the inference cache so it reloads
    try:
        from src.phoneme_inference import reset
        reset()
    except ImportError:
        pass


def show_info(name: str):
    """Show detailed metadata for a model version."""
    path = _version_path(name)
    if not os.path.isfile(path):
        # Try active model
        if name in ("active", "current"):
            path = ACTIVE_MODEL
        else:
            print(f"  Version '{name}' not found.")
            return

    try:
        data = np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"  Failed to load: {e}")
        return

    vocab = data["vocab"].tolist()
    mean_dur = float(data["mean_duration"])
    model_type = str(data["model_type"]) if "model_type" in data else "ridge"

    print(f"\n  Model: {name}")
    print(f"  Path:  {path}")
    print(f"  Type:  {model_type}")
    print(f"  {'':->50}")
    print(f"  Vocabulary ({len(vocab)} phonemes): {vocab}")
    print(f"  Mean duration: {mean_dur:.4f}s")

    if model_type == "mlp":
        for i in range(1, 4):
            W_layer = data[f"W{i}"]
            b_layer = data[f"b{i}"]
            print(f"  Layer {i}: W{i}={W_layer.shape}, b{i}={b_layer.shape}")
        n_params = sum(data[f"W{i}"].size + data[f"b{i}"].size for i in range(1, 4))
        print(f"  Parameters:  {n_params:,}")
    else:
        W = data["W"]
        b = data["b"]
        print(f"  Features:    {W.shape[1]} dims")
        print(f"  Parameters:  {W.shape[0] * W.shape[1] + W.shape[0]:,}")
        print(f"  W shape: {W.shape}, b shape: {b.shape}")

    if "metadata_json" in data:
        meta = json.loads(str(data["metadata_json"]))
        print(f"\n  Training metadata:")
        for k, v in sorted(meta.items()):
            print(f"    {k}: {v}")


def compare_versions(names: list, dataset_path: str = None):
    """
    Compare multiple model versions side by side on the same evaluation data.

    If dataset_path is given, evaluates on that dataset. Otherwise uses
    the default Azure dataset.
    """
    from src.data_format import TrainingDataset
    from src.phoneme_model import (
        build_vocabulary, featurize, extract_targets, predict, predict_mlp,
        evaluate_model
    )

    # Find dataset
    if dataset_path is None:
        ds_dir = os.path.join(_PROJECT_DIR, "data", "datasets")
        candidates = [
            os.path.join(ds_dir, "harvard_azure_disambig.json.gz"),
            os.path.join(ds_dir, "harvard_azure.json.gz"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                dataset_path = c
                break

    if not dataset_path or not os.path.isfile(dataset_path):
        print(f"  No evaluation dataset found.")
        return

    print(f"\n  Evaluation dataset: {os.path.basename(dataset_path)}")

    # Load dataset and extract pairs
    ds = TrainingDataset.load(dataset_path)
    all_pairs = ds.get_all_labeled_frames()
    print(f"  {len(all_pairs)} phoneme instances")

    # Use same val split as training (seed=42, 15%)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_pairs))
    split = int(0.85 * len(indices))
    val_idx = indices[split:]
    val_pairs = [all_pairs[i] for i in val_idx]
    print(f"  Validation set: {len(val_pairs)} instances")

    # Evaluate each model
    results = {}
    for name in names:
        path = _version_path(name)
        if not os.path.isfile(path):
            if name in ("active", "current") and os.path.isfile(ACTIVE_MODEL):
                path = ACTIVE_MODEL
            else:
                print(f"\n  Version '{name}' not found, skipping.")
                continue

        try:
            data = np.load(path, allow_pickle=False)
            model_type = str(data["model_type"]) if "model_type" in data else "ridge"
            vocab = data["vocab"].tolist()

            W, b_vec, layers = None, None, None
            if model_type == "mlp":
                layers = [
                    (data["W1"], data["b1"]),
                    (data["W2"], data["b2"]),
                    (data["W3"], data["b3"]),
                ]
            else:
                W = data["W"]
                b_vec = data["b"]
        except Exception as e:
            print(f"\n  Failed to load '{name}': {e}")
            continue

        # Featurize val pairs using this model's vocabulary
        X_val = featurize(val_pairs, vocab)
        Y_val = extract_targets(val_pairs)

        eval_result = evaluate_model(
            X_val, Y_val, W, b_vec, vocab, val_pairs,
            model_type=model_type, layers=layers,
        )
        results[name] = eval_result

    if not results:
        print(f"  No models to compare.")
        return

    # Print comparison table
    print(f"\n  {'=' * 70}")
    print(f"  MODEL COMPARISON")
    print(f"  {'=' * 70}")

    print(f"\n  {'Metric':<25}", end="")
    for name in results:
        print(f" {name:>12}", end="")
    print()
    print(f"  {'-' * (25 + 13 * len(results))}")

    # Key metrics
    metrics = [
        ("Avg mouth MAE", "avg_mouth_mae"),
        ("Avg all MAE", "avg_all_mae"),
        ("Quality score", "quality_score"),
    ]

    for label, key in metrics:
        print(f"  {label:<25}", end="")
        vals = [results[n].get(key, 0) for n in results]
        best = min(vals) if "mae" in key.lower() else max(vals)
        for name in results:
            val = results[name].get(key, 0)
            marker = " <--" if val == best and len(results) > 1 else "    "
            print(f" {val:>8.4f}{marker}", end="")
        print()

    # Per-channel comparison (top 6 mouth channels by MAE)
    first_result = list(results.values())[0]
    mouth_mae = first_result.get("mouth_mae", {})
    top_channels = sorted(mouth_mae.keys(), key=lambda x: mouth_mae[x], reverse=True)[:6]

    print(f"\n  Top channels:")
    for ch in top_channels:
        print(f"  {ch:<25}", end="")
        for name in results:
            val = results[name].get("mouth_mae", {}).get(ch, 0)
            print(f" {val:>12.4f}", end="")
        print()

    # Winner summary
    print(f"\n  {'=' * 70}")
    winner = min(results.keys(), key=lambda n: results[n]["avg_mouth_mae"])
    print(f"  BEST: {winner} (avg mouth MAE = {results[winner]['avg_mouth_mae']:.4f})")

    if winner != "active":
        print(f"\n  To activate this model:")
        print(f"    python -m src.ml.model_manager activate {winner}")


def _file_hash(path: str) -> str:
    """Quick hash for comparing files."""
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(4096))  # First 4KB is enough for identity
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Phoneme model version manager")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List all saved model versions")

    save_p = subparsers.add_parser("save", help="Save current active model as named version")
    save_p.add_argument("name", help="Version name (e.g., azure22, beat_v1)")
    save_p.add_argument("description", nargs="?", default="", help="Optional description")

    act_p = subparsers.add_parser("activate", help="Activate a named model version")
    act_p.add_argument("name", help="Version name to activate")

    info_p = subparsers.add_parser("info", help="Show model metadata")
    info_p.add_argument("name", help="Version name (or 'active')")

    cmp_p = subparsers.add_parser("compare", help="Compare model versions side by side")
    cmp_p.add_argument("names", nargs="+", help="Version names to compare")
    cmp_p.add_argument("--dataset", default=None, help="Evaluation dataset path")

    args = parser.parse_args()

    if args.command == "list":
        list_versions()
    elif args.command == "save":
        save_version(args.name, args.description)
    elif args.command == "activate":
        activate_version(args.name)
    elif args.command == "info":
        show_info(args.name)
    elif args.command == "compare":
        compare_versions(args.names, args.dataset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
