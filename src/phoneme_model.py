"""
Tier 3: Phoneme-Level Ridge Regression
========================================
Learns a direct mapping from (phoneme + context) → 55 blendshape weights,
bypassing the lossy 10-category viseme grouping.

Model: 55 independent ridge regressions (one per blendshape channel).
  input:  ~70-dim feature vector (phone one-hot + context + duration + position)
  output: 55 weights

Training data: phoneme instances extracted from gold standard dataset, each
paired with the gold-standard blendshape frame at the phoneme's midpoint.

Usage:
    python -m src.ml.phoneme_model data/ml/datasets/harvard_azure.json.gz
    python -m src.ml.phoneme_model data/ml/datasets/harvard_azure.json.gz --alpha 1.0
    python -m src.ml.phoneme_model data/ml/datasets/harvard_azure.json.gz --dry-run
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

NUM_WEIGHTS = 55
MODEL_DIR = os.path.join(_PROJECT_DIR, "data", "models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "phoneme_ridge.npz")


def build_vocabulary(pairs: list) -> list:
    """
    Build a sorted phoneme vocabulary from training data.

    Returns a list of unique phoneme strings. The index in this list
    becomes the one-hot position for that phoneme.
    """
    phones = set()
    for p in pairs:
        phones.add(p["phone"])
        phones.add(p["prev_phone"])
        phones.add(p["next_phone"])
    return sorted(phones)


def featurize(pairs: list, vocab: list, use_articulatory: bool = False) -> np.ndarray:
    """
    Convert labeled pairs to feature matrix.

    Two encoding modes:
      - One-hot (default): 3*len(vocab) + 4 dims. Each phoneme slot is a
        one-hot over the vocabulary.
      - Articulatory: 3*22 + 4 = 70 dims. Each phoneme slot encodes place,
        manner, voicing, height, backness, rounding. Encodes phoneme
        similarity so rare phonemes can borrow from neighbors.

    Both modes append: duration (1 dim) + position (3 dims one-hot).

    Returns:
        (N, D) float32 array
    """
    if use_articulatory:
        return _featurize_articulatory(pairs)
    return _featurize_onehot(pairs, vocab)


def _featurize_onehot(pairs: list, vocab: list) -> np.ndarray:
    phone_to_idx = {p: i for i, p in enumerate(vocab)}
    v = len(vocab)
    d = 3 * v + 4

    durations = [p["duration_s"] for p in pairs]
    mean_dur = np.mean(durations) if durations else 0.08

    X = np.zeros((len(pairs), d), dtype=np.float32)

    for i, p in enumerate(pairs):
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
        _encode_position(X, i, 3 * v + 1, p)

    return X


def _featurize_articulatory(pairs: list) -> np.ndarray:
    from src.articulatory import encode_triplet, TRIPLET_DIM

    durations = [p["duration_s"] for p in pairs]
    mean_dur = np.mean(durations) if durations else 0.08

    X = np.zeros((len(pairs), TRIPLET_DIM), dtype=np.float32)

    for i, p in enumerate(pairs):
        triplet = encode_triplet(p["prev_phone"], p["phone"], p["next_phone"])
        phoneme_dims = len(triplet)
        X[i, :phoneme_dims] = triplet
        X[i, phoneme_dims] = p["duration_s"] / mean_dur if mean_dur > 0 else 1.0
        _encode_position(X, i, phoneme_dims + 1, p)

    return X


def _encode_position(X: np.ndarray, row: int, offset: int, pair: dict):
    pos = pair.get("position", "mid")
    if pos == "start":
        X[row, offset] = 1.0
    elif pos == "end":
        X[row, offset + 1] = 1.0
    else:
        X[row, offset + 2] = 1.0


def extract_targets(pairs: list) -> np.ndarray:
    """Extract target weight matrix from labeled pairs. Returns (N, 55)."""
    Y = np.zeros((len(pairs), NUM_WEIGHTS), dtype=np.float32)
    for i, p in enumerate(pairs):
        weights = p["target_weights"]
        Y[i, :len(weights)] = weights[:NUM_WEIGHTS]
    return Y


def train_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0,
                sample_weights: np.ndarray = None) -> tuple:
    """
    Train 55 ridge regressions (one per blendshape weight).

    Closed-form solution: W = (X^T W_diag X + alpha I)^{-1} X^T W_diag Y

    Args:
        X: Feature matrix (N, D)
        Y: Target matrix (N, 55)
        alpha: Regularization strength (higher = more regularization)
        sample_weights: Per-sample weights (N,). If None, uniform weighting.

    Returns:
        (W, b) where W is (55, D) and b is (55,)
    """
    N, D = X.shape

    # Add bias column
    X_bias = np.hstack([X, np.ones((N, 1), dtype=np.float32)])
    D_bias = D + 1

    if sample_weights is not None:
        sqrt_w = np.sqrt(sample_weights).astype(np.float32)
        X_w = X_bias * sqrt_w[:, None]
        Y_w = Y * sqrt_w[:, None]
    else:
        X_w = X_bias
        Y_w = Y

    XtX = X_w.T @ X_w
    regularizer = alpha * np.eye(D_bias, dtype=np.float32)
    regularizer[-1, -1] = 0.0  # Don't regularize the bias term

    W_full = np.linalg.solve(XtX + regularizer, X_w.T @ Y_w)  # (D+1, 55)

    W = W_full[:-1, :].T  # (55, D)
    b = W_full[-1, :]      # (55,)

    return W, b


def compute_sample_weights(pairs: list) -> np.ndarray:
    """Inverse-frequency weights so rare phonemes contribute equally."""
    from collections import Counter
    counts = Counter(p["phone"] for p in pairs)
    return np.array([1.0 / counts[p["phone"]] for p in pairs], dtype=np.float32)


def train_mlp(
    X: np.ndarray,
    Y: np.ndarray,
    hidden1: int = 128,
    hidden2: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 200,
    patience: int = 15,
    val_X: np.ndarray = None,
    val_Y: np.ndarray = None,
) -> tuple:
    """
    Train a 2-layer MLP: D -> hidden1 -> hidden2 -> 55.

    Uses PyTorch for GPU-accelerated training. Returns layer weights as
    numpy arrays (no torch dependency needed at inference time).

    Args:
        X: Feature matrix (N, D)
        Y: Target matrix (N, 55)
        hidden1: First hidden layer size
        hidden2: Second hidden layer size
        lr: Learning rate for Adam
        weight_decay: L2 regularization (like ridge alpha)
        batch_size: Mini-batch size
        max_epochs: Maximum training epochs
        patience: Early stopping patience (epochs without improvement)
        val_X: Validation features (for early stopping)
        val_Y: Validation targets

    Returns:
        List of (W, b) tuples for each layer, as numpy arrays.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = X.shape[1]

    # Model
    model = nn.Sequential(
        nn.Linear(D, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, NUM_WEIGHTS),
        nn.Sigmoid(),
    ).to(device)

    # Data
    train_ds = TensorDataset(
        torch.from_numpy(X).to(device),
        torch.from_numpy(Y).to(device),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Validation tensors
    has_val = val_X is not None and val_Y is not None
    if has_val:
        val_X_t = torch.from_numpy(val_X).to(device)
        val_Y_t = torch.from_numpy(val_Y).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), Y_batch)
            loss.backward()
            optimizer.step()

        # Early stopping on validation
        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(val_X_t), val_Y_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if (epoch + 1) % 50 == 0:
            msg = f"    Epoch {epoch + 1}/{max_epochs}"
            if has_val:
                msg += f"  val_loss={val_loss:.6f}"
            print(msg)

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Extract numpy weights
    model.cpu()
    state = model.state_dict()
    layers = [
        (state["0.weight"].numpy(), state["0.bias"].numpy()),   # (hidden1, D), (hidden1,)
        (state["2.weight"].numpy(), state["2.bias"].numpy()),   # (hidden2, hidden1), (hidden2,)
        (state["4.weight"].numpy(), state["4.bias"].numpy()),   # (55, hidden2), (55,)
    ]

    return layers


def predict_mlp(X: np.ndarray, layers: list) -> np.ndarray:
    """
    Pure numpy MLP forward pass: ReLU hidden layers, sigmoid output.

    Args:
        X: Feature matrix (N, D) or (D,) for single sample
        layers: List of (W, b) tuples from train_mlp()

    Returns:
        (N, 55) or (55,) predicted weights in [0, 1]
    """
    single = X.ndim == 1
    if single:
        X = X.reshape(1, -1)

    h = X
    for W, b in layers[:-1]:
        h = np.maximum(0, h @ W.T + b)  # ReLU
    W_out, b_out = layers[-1]
    logits = h @ W_out.T + b_out
    Y_pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))  # sigmoid

    return Y_pred[0] if single else Y_pred


def predict(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Predict blendshape weights from features.

    Args:
        X: Feature matrix (N, D) or (D,) for single sample
        W: Weight matrix (55, D)
        b: Bias vector (55,)

    Returns:
        (N, 55) or (55,) predicted weights, clipped to [0, 1]
    """
    single = X.ndim == 1
    if single:
        X = X.reshape(1, -1)

    Y_pred = X @ W.T + b
    Y_pred = np.clip(Y_pred, 0.0, 1.0)

    return Y_pred[0] if single else Y_pred


def evaluate_model(
    X_val: np.ndarray,
    Y_val: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    vocab: list,
    val_pairs: list,
    model_type: str = "ridge",
    layers: list = None,
) -> dict:
    """
    Evaluate the trained model on validation data.

    Returns per-channel MAE and per-phoneme MAE.
    For MLP models, pass model_type="mlp" and layers.
    """
    if model_type == "mlp" and layers is not None:
        Y_pred = predict_mlp(X_val, layers)
    else:
        Y_pred = predict(X_val, W, b)
    errors = np.abs(Y_val - Y_pred)

    # Per-channel MAE
    channel_mae = errors.mean(axis=0).tolist()

    # Mouth channel indices for focused reporting
    mouth_channels = {
        "jawOpen": 17, "mouthClose": 18, "mouthFunnel": 19, "mouthPucker": 20,
        "mouthSmileLeft": 23, "mouthSmileRight": 24,
        "mouthStretchLeft": 29, "mouthStretchRight": 30,
        "mouthRollLower": 31, "mouthRollUpper": 32,
        "mouthPressLeft": 35, "mouthPressRight": 36,
        "mouthLowerDownLeft": 37, "mouthLowerDownRight": 38,
        "mouthUpperUpLeft": 39, "mouthUpperUpRight": 40,
    }

    mouth_mae = {name: channel_mae[idx] for name, idx in mouth_channels.items()}
    avg_mouth_mae = np.mean([channel_mae[idx] for idx in mouth_channels.values()])

    # Per-phoneme MAE
    phoneme_mae = {}
    for i, p in enumerate(val_pairs):
        phone = p["phone"]
        if phone not in phoneme_mae:
            phoneme_mae[phone] = []
        phoneme_mae[phone].append(errors[i].mean())

    phoneme_avg_mae = {
        phone: float(np.mean(errs)) for phone, errs in phoneme_mae.items()
    }

    return {
        "channel_mae": channel_mae,
        "mouth_mae": mouth_mae,
        "avg_mouth_mae": float(avg_mouth_mae),
        "avg_all_mae": float(errors.mean()),
        "phoneme_mae": phoneme_avg_mae,
        "quality_score": float(max(0.0, 1.0 - avg_mouth_mae)),
    }


def save_model(
    path: str,
    W: np.ndarray = None,
    b: np.ndarray = None,
    vocab: list = None,
    mean_duration: float = 0.08,
    metadata: dict = None,
    layers: list = None,
    model_type: str = "ridge",
):
    """
    Save trained model to .npz file.

    The model file contains everything needed for inference:
    weights, bias, vocabulary, and normalization parameters.

    For ridge: pass W, b.
    For MLP: pass layers (list of (W, b) tuples from train_mlp).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    save_dict = {
        "model_type": np.array(model_type, dtype="U"),
        "vocab": np.array(vocab, dtype="U"),
        "mean_duration": np.float32(mean_duration),
    }

    if model_type == "mlp" and layers is not None:
        for i, (W_layer, b_layer) in enumerate(layers, start=1):
            save_dict[f"W{i}"] = W_layer
            save_dict[f"b{i}"] = b_layer
    else:
        save_dict["W"] = W
        save_dict["b"] = b

    if metadata:
        save_dict["metadata_json"] = np.array(
            json.dumps(metadata), dtype="U"
        )

    np.savez_compressed(path, **save_dict)


def load_model(path: str) -> dict:
    """
    Load trained model from .npz file.

    Returns dict with model_type, vocab, mean_duration, and either
    W/b (ridge) or layers (MLP).
    """
    data = np.load(path, allow_pickle=False)

    model_type = str(data["model_type"]) if "model_type" in data else "ridge"

    result = {
        "model_type": model_type,
        "vocab": data["vocab"].tolist(),
        "mean_duration": float(data["mean_duration"]),
    }

    if model_type == "mlp":
        result["layers"] = [
            (data["W1"], data["b1"]),
            (data["W2"], data["b2"]),
            (data["W3"], data["b3"]),
        ]
    else:
        result["W"] = data["W"]
        result["b"] = data["b"]

    if "metadata_json" in data:
        result["metadata"] = json.loads(str(data["metadata_json"]))

    return result


def train_and_save(
    dataset_path: str,
    model_path: str = None,
    alpha: float = 1.0,
    dry_run: bool = False,
    model_type: str = "ridge",
    use_articulatory: bool = False,
    multi_sample: bool = False,
    speaker_norm: bool = False,
) -> dict:
    """
    Full training pipeline: load data, split, train, evaluate, save.

    Args:
        dataset_path: Path to training dataset (.json.gz)
        model_path: Output model path (.npz). Default: data/ml/models/phoneme_ridge.npz
        alpha: Ridge regularization strength
        dry_run: If True, train and evaluate but don't save
        model_type: "ridge" for linear regression, "mlp" for neural network

    Returns:
        Dict with training results and evaluation metrics.
    """
    from src.data_format import TrainingDataset

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    label = "MLP (128->64->55)" if model_type == "mlp" else "Ridge Regression"
    print(f"\n{'=' * 60}")
    print(f"  Tier 3: Phoneme-Level {label}")
    print(f"{'=' * 60}")

    # Load dataset
    print(f"  Loading: {dataset_path}")
    ds = TrainingDataset.load(dataset_path)
    print(f"  {ds.metadata.get('num_sentences', 0)} sentences, "
          f"{ds.metadata.get('total_frames', 0)} frames")

    # Extract labeled pairs
    sample_label = "multi-sample (25/50/75%)" if multi_sample else "midpoint only"
    print(f"  Extracting labeled pairs ({sample_label})...")
    all_pairs = ds.get_all_labeled_frames(multi_sample=multi_sample)
    print(f"  {len(all_pairs)} phoneme instances")

    if len(all_pairs) < 100:
        print(f"  ERROR: Too few training examples ({len(all_pairs)})")
        return {"error": "insufficient data"}

    if speaker_norm and model_type == "mlp":
        # Sigmoid output can't produce the negative deltas speaker norm needs
        print(f"  ERROR: --speaker-norm is only supported with ridge")
        return {"error": "speaker_norm requires ridge"}

    # Build vocabulary
    vocab = build_vocabulary(all_pairs)
    print(f"  Vocabulary: {len(vocab)} phonemes: {vocab}")

    # Split: 85% train, 15% validation by sentence (no cross-sentence leakage)
    rng = np.random.RandomState(42)
    sentence_ids = sorted(set(p["sentence_idx"] for p in all_pairs))
    sentence_perm = rng.permutation(len(sentence_ids))
    split = int(0.85 * len(sentence_perm))
    train_sentences = set(sentence_ids[i] for i in sentence_perm[:split])

    train_pairs = [p for p in all_pairs if p["sentence_idx"] in train_sentences]
    val_pairs = [p for p in all_pairs if p["sentence_idx"] not in train_sentences]
    print(f"  Train: {len(train_pairs)} ({split} sentences), "
          f"Val: {len(val_pairs)} ({len(sentence_ids) - split} sentences)")

    # Featurize
    feat_label = "articulatory" if use_articulatory else "one-hot"
    print(f"  Featurizing ({feat_label})...")
    X_train = featurize(train_pairs, vocab, use_articulatory=use_articulatory)
    Y_train = extract_targets(train_pairs)
    X_val = featurize(val_pairs, vocab, use_articulatory=use_articulatory)
    Y_val = extract_targets(val_pairs)

    mean_duration = float(np.mean([p["duration_s"] for p in train_pairs]))

    D = X_train.shape[1]
    print(f"  Features: {D} dims ({len(vocab)}*3 one-hot + 1 duration + 3 position)")

    # Train
    W, b, layers = None, None, None

    if model_type == "mlp":
        total_params = 128 * D + 128 + 64 * 128 + 64 + 55 * 64 + 55
        print(f"  Parameters: {total_params:,}")
        print(f"  Data/param ratio: {len(train_pairs) / total_params:.1f}:1")
        print(f"  Training MLP...")
        t0 = time.time()
        layers = train_mlp(X_train, Y_train, val_X=X_val, val_Y=Y_val)
        elapsed = time.time() - t0
        print(f"  Trained in {elapsed:.1f}s")
    else:
        print(f"  Parameters: {55 * D:,} (55 channels x {D} features)")
        print(f"  Data/param ratio: {len(train_pairs) / (55 * D):.1f}:1")
        sample_w = compute_sample_weights(train_pairs)
        Y_fit = Y_train
        if speaker_norm:
            # Fit deltas from each speaker's mean (train split only, so no
            # val leakage), then fold the average neutral face into the bias
            # so the saved model outputs absolute weights for unseen speakers.
            from src.speaker_norm import compute_speaker_means, speaker_offsets
            speaker_means = compute_speaker_means(train_pairs)
            neutral = np.mean(list(speaker_means.values()), axis=0)
            Y_fit = Y_train - speaker_offsets(train_pairs, speaker_means, neutral)
            print(f"  Speaker normalization: {len(speaker_means)} train speakers, "
                  f"fitting deltas, neutral face folded into bias")
        print(f"  Training (alpha={alpha}, inverse-frequency weighted)...")
        t0 = time.time()
        W, b = train_ridge(X_train, Y_fit, alpha=alpha, sample_weights=sample_w)
        if speaker_norm:
            b = b + neutral
        elapsed = time.time() - t0
        print(f"  Trained in {elapsed:.3f}s")
        print(f"  W shape: {W.shape}, b shape: {b.shape}")

    # Evaluate on validation set
    print(f"\n  Evaluating on validation set...")
    eval_result = evaluate_model(
        X_val, Y_val, W, b, vocab, val_pairs,
        model_type=model_type, layers=layers,
    )

    # Print evaluation
    print(f"\n  {'Channel':<25} {'MAE':>8}")
    print(f"  {'-' * 35}")
    for name, mae in sorted(eval_result["mouth_mae"].items(),
                            key=lambda x: eval_result["mouth_mae"][x[0]], reverse=True):
        flag = " <<<" if mae > 0.05 else ""
        print(f"  {name:<25} {mae:>8.4f}{flag}")

    print(f"\n  Avg mouth MAE:   {eval_result['avg_mouth_mae']:.4f}")
    print(f"  Avg all MAE:     {eval_result['avg_all_mae']:.4f}")
    print(f"  Quality score:   {eval_result['quality_score']:.3f}")

    # Per-phoneme breakdown
    print(f"\n  Per-phoneme MAE:")
    print(f"  {'Phone':<8} {'MAE':>8} {'N':>6}")
    print(f"  {'-' * 25}")
    phone_counts = {}
    for p in val_pairs:
        phone_counts[p["phone"]] = phone_counts.get(p["phone"], 0) + 1

    for phone, mae in sorted(eval_result["phoneme_mae"].items(),
                              key=lambda x: x[1], reverse=True):
        n = phone_counts.get(phone, 0)
        print(f"  {phone:<8} {mae:>8.4f} {n:>6}")

    # Also evaluate on training set (to check for underfitting)
    train_eval = evaluate_model(
        X_train, Y_train, W, b, vocab, train_pairs,
        model_type=model_type, layers=layers,
    )
    print(f"\n  Train avg mouth MAE: {train_eval['avg_mouth_mae']:.4f}")
    gap = eval_result["avg_mouth_mae"] - train_eval["avg_mouth_mae"]
    print(f"  Train/val gap:       {gap:+.4f} ({'overfitting' if gap > 0.01 else 'good'})")

    # Save
    if dry_run:
        print(f"\n  DRY RUN -- model not saved")
    else:
        metadata = {
            "model_type": model_type,
            "feature_encoding": "articulatory" if use_articulatory else "onehot",
            "speaker_norm": speaker_norm,
            "n_train": len(train_pairs),
            "n_val": len(val_pairs),
            "n_features": D,
            "vocab_size": len(vocab),
            "val_quality_score": eval_result["quality_score"],
            "val_avg_mouth_mae": eval_result["avg_mouth_mae"],
            "dataset": os.path.basename(dataset_path),
        }
        if model_type == "ridge":
            metadata["alpha"] = alpha

        save_model(
            model_path, W=W, b=b, vocab=vocab,
            mean_duration=mean_duration, metadata=metadata,
            layers=layers, model_type=model_type,
        )
        print(f"\n  Model saved: {model_path}")

    print(f"{'=' * 60}")

    return {
        "model_path": model_path if not dry_run else None,
        "vocab": vocab,
        "n_features": D,
        "train_eval": train_eval,
        "val_eval": eval_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Tier 3 phoneme-level viseme model"
    )
    parser.add_argument("dataset", help="Path to training dataset (.json.gz)")
    parser.add_argument("--output", default=None, help="Output model path (.npz)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularization strength (default: 1.0)")
    parser.add_argument("--model-type", choices=["ridge", "mlp"], default="ridge",
                        help="Model type: ridge (linear) or mlp (neural network)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Train and evaluate without saving")
    parser.add_argument("--articulatory", action="store_true",
                        help="Use articulatory features instead of one-hot encoding")
    parser.add_argument("--multi-sample", action="store_true",
                        help="Sample at 25/50/75%% of phoneme duration instead of midpoint only")
    parser.add_argument("--speaker-norm", action="store_true",
                        help="Per-speaker mean subtraction (predict deltas from neutral)")
    args = parser.parse_args()

    train_and_save(
        dataset_path=args.dataset,
        model_path=args.output,
        alpha=args.alpha,
        dry_run=args.dry_run,
        model_type=args.model_type,
        use_articulatory=args.articulatory,
        multi_sample=args.multi_sample,
        speaker_norm=args.speaker_norm,
    )


if __name__ == "__main__":
    main()
