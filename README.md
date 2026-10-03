# Viseme Training

Trains ML models to convert phoneme timing from TTS providers into 55 ARKit blendshape weights for avatar lip-sync.

## The Problem

Most TTS providers (Inworld AI, Azure, ElevenLabs, etc.) give you phoneme-level timing alongside audio, but they don't give you visemes, which are the actual mouth shapes your avatar needs. That gap is usually filled by open-source tools like [Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync) or [Oculus Lipsync](https://developer.oculus.com/documentation/unity/audio-ovrlipsync-unity/), which map phonemes to a handful of static mouth poses.

Those work okay, but they aren't natural enough. They don't account for how surrounding phonemes shape each mouth position (coarticulation) and output maybe 10-15 categories, not the full 55 ARKit weights an avatar rig actually uses.

This project trains a model that takes phoneme sequences with context (previous/next phoneme, duration, position) and predicts all 55 blendshape weights per phoneme.

## What I Learned (the hard way)

I started by training against Azure TTS's blendshape output as "ground truth." The model scored 0.955 quality, looks great right? But it looked terrible on the avatar. Turns out Azure's output has compressed dynamic range, missing bilabial closure and groups 40 phonemes into 22 categories. So I was 95.5% accurate at being wrong.

Switching to BEAT v1 (real iPhone ARKit captures from 30 speakers, ~60 hours of English speech) helped with the quality. The current best model (`phoneme_ridge_beat.npz`) scores 0.923 on held-out data and actually looks right.

Full writeup of the mistakes I made, so you don't do the same: [`docs/LEARNING_JOURNAL.md`](docs/LEARNING_JOURNAL.md)
Known limitations and TODO backlog: [`docs/BIAS_ANALYSIS.md`](docs/BIAS_ANALYSIS.md)

## Architecture

```
ARPAbet phoneme sequence  ─┐
   (current + prev + next) ├──► Featurization ──► Model ──► 55 ARKit
       phoneme duration   ─┤    (3V + 4 dims)    (ridge   blendshape
       sentence position  ─┘                      or MLP)   weights
```

The trained `.npz` model is self-contained — `numpy.load(path)` gives you weights, vocabulary, mean phoneme duration, and training metadata.

## Quick Start

```bash
# Train on BEAT v1 (after downloading — see below)
python -m src.phoneme_model data/datasets/beat_v1.json.gz \
    --output data/models/phoneme_ridge_beat.npz

# Compare models side-by-side
python -m src.model_manager compare azure22 azure39 beat
```

## Reproducing (BEAT v1)

```bash
# 1. Download BEAT v1 from HuggingFace
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='H-Liu1997/BEAT', repo_type='dataset', local_dir='./BEAT',
                  allow_patterns=['*/*.json', '*/*.wav', '*/*.TextGrid'])
"

# 2. Build the training dataset
python -m src.beat_adapter /path/to/BEAT/beat_english_v0.2.1/beat_english_v0.2.1 --build-dataset

# 3. Train
python -m src.phoneme_model data/datasets/beat_v1.json.gz \
    --output data/models/phoneme_ridge_beat.npz

# OR use the streaming loader if RAM is tight (~2-4 GB instead of ~12 GB)
python main.py --dataset data/datasets/beat_v1.json.gz
```

### Optional: MFA Phoneme Alignment

BEAT's TextGrids already have phoneme timing from iPhone capture. If you want better alignments via Montreal Forced Aligner:

```bash
conda create -n mfa -c conda-forge montreal-forced-aligner python=3.11
conda activate mfa
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
python -m src.mfa_aligner /path/to/BEAT --output data/mfa_alignments
conda deactivate

python -m src.beat_adapter /path/to/BEAT --build-dataset --mfa-dir data/mfa_alignments/alignments
```

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
| `src/gold_standard.py` | Provider interface + Azure / BEAT adapters |
| `src/beat_adapter.py` | BEAT v1 data loader (52→55 weights, 60→30 FPS) |
| `src/collect_data.py` | Collect Azure gold standard data (see learning journal for why this was a dead end) |
| `src/cmudict_disambiguator.py` | Expand Azure 22 viseme IDs → 39 phonemes |
| `src/mfa_aligner.py` | Montreal Forced Aligner wrapper |
| `src/evaluation.py` | MAE, RMSE, correlation, velocity, DTW, hit-rate |
| `src/model_manager.py` | Version management (save / list / compare / activate) |
| `src/statistical_analysis.py` | Per-category variance, PCA, split recommendations |
| `src/arpabet.py` | ARPAbet phoneme-to-viseme mapping constants |
| `src/harvard_sentences.py` | 720 IEEE sentences + train/val/test splits |
| `main.py` | Streaming MLP training entry point (low RAM) |
| `train_mlp.py` | In-memory MLP training script |
| `train_mlp.ipynb` | Notebook walkthrough: ridge baseline + MLP comparison |

## Acknowledgments

- **BEAT v1** — Liu et al., "BEAT: A Large-Scale Semantic and Emotional Multi-Modal Dataset for Conversational Gestures Synthesis" (ECCV 2022). Licensed Apache-2.0. Dataset: <https://huggingface.co/datasets/H-Liu1997/BEAT>.
- **CMUdict** — for phoneme disambiguation.

## License

MIT — see `LICENSE`.
