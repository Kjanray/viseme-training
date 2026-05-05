"""
MLP Viseme Model Training — Streaming Data Loader
===================================================
Memory-efficient version of train_mlp.py that streams the JSON dataset
sentence-by-sentence instead of loading all 6.7M frames into memory at once.

Reduces peak RAM from ~12GB to ~2-4GB for the beat_v1 dataset.

Requires: pip install ijson tqdm

Usage:
    cd viseme-training
    pip install ijson tqdm
    python main.py
    python main.py --dataset data/datasets/beat_v1.json.gz
"""

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import ijson
except ImportError:
    print("ERROR: ijson is required for streaming JSON parsing.")
    print("  Install with: pip install ijson")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "data", "datasets")

NUM_WEIGHTS = 55

DATASET_CANDIDATES = [
    os.path.join(_DATA_DIR, "beat_v1.json.gz"),
    os.path.join(_DATA_DIR, "harvard_azure_disambig.json.gz"),
    os.path.join(_DATA_DIR, "harvard_azure.json.gz"),
]

# ---------------------------------------------------------------------------
# Streaming data loading
# ---------------------------------------------------------------------------

def _find_nearest_frame(frame_times, target_ms):
    """Binary search for nearest frame to target_ms."""
    if not frame_times:
        return None
    lo, hi = 0, len(frame_times) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if frame_times[mid][0] < target_ms:
            lo = mid + 1
        else:
            hi = mid
    best_idx = lo
    if lo > 0 and abs(frame_times[lo - 1][0] - target_ms) < abs(frame_times[lo][0] - target_ms):
        best_idx = lo - 1
    return frame_times[best_idx][1]


def _extract_pairs_from_sentence(sentence):
    """Extract (phoneme, target_weights) pairs from a single sentence."""
    pairs = []
    timeline = sentence.get("phoneme_timeline", [])
    frames = sentence.get("target_frames", [])
    if not timeline or not frames:
        return pairs

    # Convert frames to (time_ms, weights) tuples — only this sentence's frames
    # Cast to float: ijson returns decimal.Decimal for JSON numbers
    frame_times = [(float(f["time_ms"]), [float(w) for w in f["weights"]]) for f in frames]

    for i, phoneme in enumerate(timeline):
        phone = phoneme.get("phone", "sil")
        start_s = phoneme.get("start_s", 0.0)
        duration_s = phoneme.get("duration_s", 0.0)
        start_s = float(start_s)
        duration_s = float(duration_s)
        mid_ms = (start_s + duration_s / 2.0) * 1000.0
        nearest_weights = _find_nearest_frame(frame_times, mid_ms)
        if nearest_weights is None:
            continue
        prev_phone = timeline[i - 1].get("phone", "sil") if i > 0 else "sil"
        next_phone = timeline[i + 1].get("phone", "sil") if i < len(timeline) - 1 else "sil"
        if i == 0:
            position = "start"
        elif i == len(timeline) - 1:
            position = "end"
        else:
            position = "mid"
        pairs.append({
            "phone": phone,
            "prev_phone": prev_phone,
            "next_phone": next_phone,
            "duration_s": duration_s,
            "position": position,
            "target_weights": nearest_weights,
        })
    return pairs


def read_metadata(path: str) -> dict:
    """Read only the metadata object from a dataset (fast, small)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        for item in ijson.items(f, "metadata"):
            return item
    return {}


def stream_labeled_frames(path: str) -> list:
    """
    Stream a .json.gz dataset sentence-by-sentence using ijson.
    Only one sentence's frames are in memory at a time.

    Returns the list of all phoneme pairs (much smaller than raw frames).
    """
    all_pairs = []

    opener = gzip.open if path.endswith(".gz") else open

    # Read total sentence count from metadata for tqdm progress bar
    total_sentences = None
    try:
        meta = read_metadata(path)
        total_sentences = meta.get("num_sentences")
    except Exception:
        pass

    t0 = time.time()
    with opener(path, "rb") as f:
        pbar = tqdm(ijson.items(f, "sentences.item"),
                     total=total_sentences,
                     desc="  Streaming sentences",
                     unit="sent",
                     ncols=90,
                     leave=True)
        for sentence in pbar:
            pairs = _extract_pairs_from_sentence(sentence)
            all_pairs.extend(pairs)
            pbar.set_postfix(pairs=len(all_pairs), refresh=False)

    elapsed = time.time() - t0
    tqdm.write(f"  Streamed -> {len(all_pairs)} pairs in {elapsed:.1f}s")

    return all_pairs


# ---------------------------------------------------------------------------
# Featurization (same as train_mlp.py)
# ---------------------------------------------------------------------------

def build_vocabulary(pairs):
    phones = set()
    for p in pairs:
        phones.add(p["phone"])
        phones.add(p["prev_phone"])
        phones.add(p["next_phone"])
    return sorted(phones)


def featurize(pairs, vocab):
    phone_to_idx = {p: i for i, p in enumerate(vocab)}
    v = len(vocab)
    d = 3 * v + 4
    durations = [p["duration_s"] for p in pairs]
    mean_dur = float(np.mean(durations)) if durations else 0.08

    X = np.zeros((len(pairs), d), dtype=np.float32)
    for i, p in enumerate(tqdm(pairs, desc="  Featurizing", unit="pair", ncols=90)):
        idx = phone_to_idx.get(p["phone"])
        if idx is not None:
            X[i, idx] = 1.0
        idx = phone_to_idx.get(p["prev_phone"])
        if idx is not None:
            X[i, v + idx] = 1.0
        idx = phone_to_idx.get(p["next_phone"])
        if idx is not None:
            X[i, 2 * v + idx] = 1.0
        X[i, 3 * v] = p["duration_s"] / mean_dur if mean_dur > 0 else 1.0
        if p["position"] == "start":
            X[i, 3 * v + 1] = 1.0
        elif p["position"] == "end":
            X[i, 3 * v + 2] = 1.0
        else:
            X[i, 3 * v + 3] = 1.0
    return X, mean_dur


def extract_targets(pairs):
    Y = np.zeros((len(pairs), NUM_WEIGHTS), dtype=np.float32)
    for i, p in enumerate(pairs):
        w = p["target_weights"]
        Y[i, :len(w)] = w[:NUM_WEIGHTS]
    return Y


# ---------------------------------------------------------------------------
# Evaluation (same as train_mlp.py)
# ---------------------------------------------------------------------------

MOUTH_CHANNELS = {
    "jawOpen": 17, "mouthClose": 18, "mouthFunnel": 19, "mouthPucker": 20,
    "mouthSmileLeft": 23, "mouthSmileRight": 24,
    "mouthStretchLeft": 29, "mouthStretchRight": 30,
    "mouthRollLower": 31, "mouthRollUpper": 32,
    "mouthPressLeft": 35, "mouthPressRight": 36,
    "mouthLowerDownLeft": 37, "mouthLowerDownRight": 38,
    "mouthUpperUpLeft": 39, "mouthUpperUpRight": 40,
}


def evaluate(Y_true, Y_pred, label=""):
    errors = np.abs(Y_true - Y_pred)
    channel_mae = errors.mean(axis=0)
    mouth_maes = {name: float(channel_mae[idx]) for name, idx in MOUTH_CHANNELS.items()}
    avg_mouth_mae = float(np.mean([channel_mae[idx] for idx in MOUTH_CHANNELS.values()]))
    avg_all_mae = float(errors.mean())
    quality = max(0.0, 1.0 - avg_mouth_mae)

    if label:
        print(f"\n  {label}:")
        print(f"    Avg mouth MAE: {avg_mouth_mae:.4f}")
        print(f"    Avg all MAE:   {avg_all_mae:.4f}")
        print(f"    Quality score: {quality:.3f}")

    return {
        "avg_mouth_mae": avg_mouth_mae,
        "avg_all_mae": avg_all_mae,
        "quality_score": quality,
        "channel_mae": channel_mae,
        "mouth_maes": mouth_maes,
    }


# ---------------------------------------------------------------------------
# MLP model (same as train_mlp.py)
# ---------------------------------------------------------------------------

class VisemeMLP(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, output_dim=55):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, output_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train MLP viseme model (streaming data loader)")
    parser.add_argument("--dataset", default=None, help="Path to .json.gz dataset")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", default=os.path.join(_SCRIPT_DIR, "phoneme_mlp.npz"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Find dataset ──────────────────────────────────────────────────────
    dataset_path = args.dataset
    if dataset_path is None:
        for candidate in DATASET_CANDIDATES:
            if os.path.isfile(candidate):
                dataset_path = candidate
                break
    if dataset_path is None:
        print("ERROR: No dataset found. Expected one of:")
        for c in DATASET_CANDIDATES:
            print(f"  {c}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  MLP Viseme Model Training (Streaming)")
    print(f"{'=' * 60}")

    # ── Load metadata (fast, small) ──────────────────────────────────────
    print(f"\n  Reading metadata...")
    meta = read_metadata(dataset_path)
    print(f"  Provider: {meta.get('provider', '?')}, "
          f"Sentences: {meta.get('num_sentences', 0)}, "
          f"Frames: {meta.get('total_frames', 0)}")

    # ── Stream data sentence-by-sentence ──────────────────────────────────
    print(f"\n  Streaming: {dataset_path}")
    all_pairs = stream_labeled_frames(dataset_path)
    print(f"  Extracted {len(all_pairs)} phoneme instances")

    vocab = build_vocabulary(all_pairs)
    print(f"  Vocabulary: {len(vocab)} phonemes")

    # ── Split & featurize ─────────────────────────────────────────────────
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_pairs))
    split = int(0.85 * len(indices))
    train_pairs = [all_pairs[i] for i in indices[:split]]
    val_pairs = [all_pairs[i] for i in indices[split:]]

    # Free the full pairs list — we only need the splits now
    del all_pairs

    X_train, mean_duration = featurize(train_pairs, vocab)
    Y_train = extract_targets(train_pairs)
    X_val, _ = featurize(val_pairs, vocab)
    Y_val = extract_targets(val_pairs)
    D = X_train.shape[1]

    # Free raw pairs — we have numpy arrays now
    del train_pairs, val_pairs

    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    print(f"  Features: {D} dims")

    # ── Ridge baseline ────────────────────────────────────────────────────
    print(f"\n  Training ridge baseline...")
    N = X_train.shape[0]
    X_bias = np.hstack([X_train, np.ones((N, 1), dtype=np.float32)])
    reg = np.eye(D + 1, dtype=np.float32)
    reg[-1, -1] = 0.0
    W_full = np.linalg.solve(X_bias.T @ X_bias + reg, X_bias.T @ Y_train)
    W_ridge, b_ridge = W_full[:-1, :].T, W_full[-1, :]

    Y_pred_ridge_val = np.clip(X_val @ W_ridge.T + b_ridge, 0, 1)
    ridge_val = evaluate(Y_val, Y_pred_ridge_val, "Ridge (val)")

    # ── MLP training ──────────────────────────────────────────────────────
    model = VisemeMLP(input_dim=D).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  MLP: {D} -> 128 -> 64 -> 55  ({total_params:,} params)")

    train_ds = TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(Y_train).to(device),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    X_val_t = torch.from_numpy(X_val).to(device)
    Y_val_t = torch.from_numpy(Y_val).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(f"  Training: lr={args.lr}, batch={args.batch_size}, patience={args.patience}")

    epoch_pbar = tqdm(range(args.epochs), desc="  Training", unit="epoch", ncols=90)
    t0 = time.time()
    for epoch in epoch_pbar:
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            vp = model(X_val_t)
            val_loss = criterion(vp, Y_val_t).item()
            val_mae = torch.abs(Y_val_t - vp).mean().item()

        train_losses.append(avg_train)
        val_losses.append(val_loss)

        epoch_pbar.set_postfix(train=f"{avg_train:.6f}", val=f"{val_loss:.6f}",
                                mae=f"{val_mae:.4f}", refresh=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                epoch_pbar.close()
                tqdm.write(f"  Early stop at epoch {epoch+1} "
                           f"(val_loss={val_loss:.6f}, val_mae={val_mae:.4f})")
                break

    elapsed = time.time() - t0
    print(f"\n  Trained in {elapsed:.1f}s ({epoch + 1} epochs)")

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    # ── Evaluation ────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        Y_pred_mlp_val = model(X_val_t).cpu().numpy()

    mlp_val = evaluate(Y_val, Y_pred_mlp_val, "MLP (val)")

    print(f"\n  {'=' * 55}")
    print(f"  COMPARISON")
    print(f"  {'=' * 55}")
    print(f"  {'Metric':<25} {'Ridge':>10} {'MLP':>10} {'Change':>10}")
    print(f"  {'-' * 55}")
    for label, key in [("Avg mouth MAE", "avg_mouth_mae"),
                        ("Avg all MAE", "avg_all_mae"),
                        ("Quality score", "quality_score")]:
        r, m = ridge_val[key], mlp_val[key]
        pct = ((m - r) / r) * 100 if r != 0 else 0
        print(f"  {label:<25} {r:>10.4f} {m:>10.4f} {pct:>+9.1f}%")

    print(f"\n  Per channel:")
    print(f"  {'Channel':<25} {'Ridge':>10} {'MLP':>10} {'Change':>10}")
    print(f"  {'-' * 55}")
    for name in sorted(MOUTH_CHANNELS.keys()):
        r = ridge_val["mouth_maes"][name]
        m = mlp_val["mouth_maes"][name]
        pct = ((m - r) / r) * 100 if r != 0 else 0
        marker = " <<<" if pct < -10 else ""
        print(f"  {name:<25} {r:>10.4f} {m:>10.4f} {pct:>+9.1f}%{marker}")

    # ── Visualization ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(train_losses, label="Train", alpha=0.8)
    ax.plot(val_losses, label="Val", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ch_sorted = sorted(MOUTH_CHANNELS.keys(),
                        key=lambda c: ridge_val["mouth_maes"][c], reverse=True)[:10]
    x_pos = np.arange(len(ch_sorted))
    w = 0.35
    ax.barh(x_pos + w/2, [ridge_val["mouth_maes"][c] for c in ch_sorted],
            w, label="Ridge", color="#4C72B0")
    ax.barh(x_pos - w/2, [mlp_val["mouth_maes"][c] for c in ch_sorted],
            w, label="MLP", color="#DD8452")
    ax.set_yticks(x_pos)
    ax.set_yticklabels(ch_sorted, fontsize=8)
    ax.set_xlabel("MAE")
    ax.set_title("Per-Channel MAE")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    for idx_plot, (ch_name, ch_idx) in enumerate([("jawOpen", 17), ("mouthFunnel", 19)]):
        ax = axes[1, idx_plot]
        ax.scatter(Y_val[:, ch_idx], Y_pred_ridge_val[:, ch_idx],
                    alpha=0.15, s=5, label="Ridge", color="#4C72B0")
        ax.scatter(Y_val[:, ch_idx], Y_pred_mlp_val[:, ch_idx],
                    alpha=0.15, s=5, label="MLP", color="#DD8452")
        ax.plot([0, 0.5], [0, 0.5], "k--", alpha=0.3, label="Perfect")
        ax.set_xlabel("Gold")
        ax.set_ylabel("Predicted")
        ax.set_title(ch_name)
        ax.legend(markerscale=4)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(_SCRIPT_DIR, "mlp_vs_ridge.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n  Saved plot: {plot_path}")

    # ── Export .npz ───────────────────────────────────────────────────────
    state = model.cpu().state_dict()
    W1, b1 = state["net.0.weight"].numpy(), state["net.0.bias"].numpy()
    W2, b2 = state["net.2.weight"].numpy(), state["net.2.bias"].numpy()
    W3, b3 = state["net.4.weight"].numpy(), state["net.4.bias"].numpy()

    metadata = {
        "model_type": "mlp",
        "architecture": [D, 128, 64, NUM_WEIGHTS],
        "learning_rate": args.lr,
        "weight_decay": 1e-4,
        "batch_size": args.batch_size,
        "epochs_trained": epoch + 1,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": D,
        "vocab_size": len(vocab),
        "val_quality_score": mlp_val["quality_score"],
        "val_avg_mouth_mae": mlp_val["avg_mouth_mae"],
        "dataset": os.path.basename(dataset_path),
    }

    np.savez_compressed(
        args.output,
        model_type=np.array("mlp", dtype="U"),
        W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
        vocab=np.array(vocab, dtype="U"),
        mean_duration=np.float32(mean_duration),
        metadata_json=np.array(json.dumps(metadata), dtype="U"),
    )

    size_kb = os.path.getsize(args.output) / 1024
    print(f"\n  Saved model: {args.output} ({size_kb:.1f} KB)")
    print(f"  Quality: {mlp_val['quality_score']:.3f} "
          f"(mouth MAE: {mlp_val['avg_mouth_mae']:.4f})")
    print(f"\n  Deploy: copy {args.output} into your inference project's model directory")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()