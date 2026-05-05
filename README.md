# Viseme Training

Offline training pipeline for phoneme-to-blendshape models that drive ARKit-style lip-sync animation. Trains on real iPhone facial capture from 30 speakers (BEAT v1, ~60 hours of English speech) to predict 55 ARKit blendshape weights from ARPAbet phoneme sequences. The current best model (`phoneme_ridge_beat.npz`) reaches a quality score of 0.923 on held-out validation data.

## What this codebase shows

- **End-to-end ML workflow:** data collection (Azure Cognitive Services TTS + Harvard sentences), label cleaning (CMUdict-based viseme disambiguation, 22 → 39 phonemes), feature engineering (one-hot phoneme + ±1 context + duration + sentence-position), model training (ridge regression baseline → 2-layer MLP), and evaluation (MAE, RMSE, DTW, per-channel diagnostics, hit-rate).
- **Two production-grade dataset adapters** behind a single provider interface (`src/gold_standard.py`): one for synthetic Azure TTS output, one for real ARKit captures from BEAT v1.
- **Memory-efficient streaming loader** (`main.py`) that processes the 632 MB BEAT dataset sentence-by-sentence, dropping peak RAM from ~12 GB to ~2-4 GB.
- **Honest failure documentation:** `docs/BIAS_ANALYSIS.md` lists every known bias and limitation in the current model, with effort estimates to address each. `docs/LEARNING_JOURNAL.md` is a postmortem-style writeup of the engineering mistakes I made and what I'd do differently.
- **Versioned model management** (`src/model_manager.py`) with side-by-side metric comparison, so any new architecture / dataset / hyperparameter change can be evaluated against the existing baselines.

## Architecture

```
ARPAbet phoneme sequence  ─┐
   (current + prev + next) ├──► Featurization ──► Model ──► 55 ARKit
       phoneme duration   ─┤    (3V + 4 dims)    (ridge   blendshape
       sentence position  ─┘                      or MLP)   weights
```

The output `.npz` model is self-contained — load it with `numpy.load(path)` and you get weights, vocabulary, mean phoneme duration, and a `metadata_json` blob describing how it was trained.

## Quick Start

```bash
# Train on the BEAT dataset (after fetching it — see "How to Reproduce")
python -m src.phoneme_model data/datasets/beat_v1.json.gz \
    --output data/models/phoneme_ridge_beat.npz

# Compare existing models side-by-side
python -m src.model_manager compare azure22 azure39 beat
```

## How to Reproduce

### Option A: Azure Gold Standard (synthetic, ~$0.44 in API costs)

```bash
# 1. Collect data (720 Harvard sentences via Azure TTS)
python -m src.collect_data --dry-run    # preview the cost
python -m src.collect_data              # actually collect

# 2. Disambiguate phoneme labels (Azure's 22 viseme IDs → 39 ARPAbet phonemes)
python -m src.cmudict_disambiguator data/datasets/harvard_azure.json.gz

# 3. Train
python -m src.phoneme_model data/datasets/harvard_azure_disambig.json.gz \
    --output data/models/phoneme_ridge_azure39.npz
```

### Option B: BEAT Gold Standard (real ARKit capture, ~60 GB download)

```bash
# 1. Download BEAT v1 from HuggingFace (filter to needed file types only)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='H-Liu1997/BEAT', repo_type='dataset', local_dir='./BEAT',
                  allow_patterns=['*/*.json', '*/*.wav', '*/*.TextGrid'])
"

# 2. (Optional) Clean leftover files if you downloaded everything
python scripts/clean_beat.py /path/to/BEAT/beat_english_v0.2.1/beat_english_v0.2.1 --delete

# 3. Build the training dataset
python -m src.beat_adapter /path/to/BEAT/beat_english_v0.2.1/beat_english_v0.2.1 --build-dataset

# 4. Train (in-memory)
python -m src.phoneme_model data/datasets/beat_v1.json.gz \
    --output data/models/phoneme_ridge_beat.npz

# OR train with the streaming MLP loader (low RAM)
python main.py --dataset data/datasets/beat_v1.json.gz
```

### Optional: MFA Phoneme Alignment

BEAT's TextGrids already include phoneme timing from iPhone capture. If you want higher-quality alignments via Montreal Forced Aligner:

```bash
conda create -n mfa -c conda-forge montreal-forced-aligner python=3.11
conda activate mfa
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
python -m src.mfa_aligner /path/to/BEAT --output data/mfa_alignments
conda deactivate

python -m src.beat_adapter /path/to/BEAT --build-dataset --mfa-dir data/mfa_alignments/alignments
```

## What I Learned

`docs/LEARNING_JOURNAL.md` is the long version. The headline lessons:

- **Train-serve skew is silent and expensive.** I spent days chasing a "model regression" that turned out to be the inference path featurizing inputs slightly differently from training. Lesson: always run a small set of production inputs through the full inference pipeline as part of the training validation step, not as a separate manual check.
- **Gold standards aren't ground truth.** Training against Azure TTS's neural blendshape output looks like supervision, but you're really learning to imitate one company's rendering choices, not the underlying articulatory physics. Switching to BEAT (real iPhone ARKit capture) was the single biggest quality jump in the whole project.
- **Linear models smooth aggressively.** Ridge regression's regression-to-mean problem (CodeTalker, CVPR 2023) is fundamental, not a tuning issue. Moving to a small MLP captured non-linear coarticulation and gave a measurable per-channel MAE drop on mouth weights.

`docs/BIAS_ANALYSIS.md` lists every known limitation currently in the pipeline plus an effort estimate per fix.

## Model Management

```bash
python -m src.model_manager list                         # All saved versions
python -m src.model_manager save my_name "description"   # Save current
python -m src.model_manager compare azure22 azure39 beat # Side-by-side metrics
python -m src.model_manager info azure39                 # Show metadata
```

## File Reference

| File | Purpose |
|------|---------|
| `src/phoneme_model.py` | Train ridge regression: phoneme + context → 55 weights |
| `src/data_format.py` | TrainingDataset class (provider-agnostic save/load) |
| `src/harvard_sentences.py` | 720 IEEE sentences + train/val/test splits |
| `src/gold_standard.py` | Provider interface + Azure / BEAT adapters |
| `src/collect_data.py` | Collect Azure gold standard data |
| `src/cmudict_disambiguator.py` | Expand Azure 22 viseme IDs → 39 phonemes |
| `src/beat_adapter.py` | BEAT v1 data loader (52→55 weights, 60→30 FPS) |
| `src/mfa_aligner.py` | Montreal Forced Aligner wrapper |
| `src/statistical_analysis.py` | Per-category variance, PCA, split recommendations |
| `src/evaluation.py` | MAE, RMSE, correlation, velocity, DTW, hit-rate |
| `src/model_manager.py` | Version management (save / list / compare / activate) |
| `src/arpabet.py` | ARPAbet phoneme-to-viseme mapping constants |
| `scripts/clean_beat.py` | Delete unneeded files from BEAT download |
| `scripts/fix_beat_download.py` | Resume incomplete BEAT downloads |
| `main.py` | Streaming MLP training entry point (low RAM) |
| `train_mlp.py` | In-memory MLP training script |
| `train_mlp.ipynb` | Notebook walkthrough: ridge baseline + MLP comparison |

## Acknowledgments

- **BEAT v1 dataset** — Liu et al., "BEAT: A Large-Scale Semantic and Emotional Multi-Modal Dataset for Conversational Gestures Synthesis" (ECCV 2022). Licensed Apache-2.0. The `phoneme_ridge_beat.npz` model in `data/models/` was trained on BEAT v1. Dataset: <https://huggingface.co/datasets/H-Liu1997/BEAT>.
- **Azure Cognitive Services Speech SDK** — used to generate the Azure gold-standard dataset (`harvard_azure*.json.gz`). The `phoneme_ridge_azure22.npz` and `phoneme_ridge_azure39.npz` models were trained on synthetic Azure TTS viseme output.
- **CMUdict** — for phoneme disambiguation in `src/cmudict_disambiguator.py`.

## License

MIT — see `LICENSE`. Model weights in `data/models/` are also released under MIT, with upstream attribution preserved via the Acknowledgments section above and the repo-root `NOTICE` file.
