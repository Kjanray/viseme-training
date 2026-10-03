# Bias Analysis & TODO Backlog

Findings from research on how phoneme-to-blendshape training is done in academia, compared with our current approach. Items are ordered by expected impact.

---

## Quick Wins (hours each)

- [x] **Sentence-level train/val split**
  - Current: random 85/15 split of individual phoneme instances
  - Problem: phonemes from the same sentence share temporal context, leaking information across the split
  - Fix: in `phoneme_model.py`, split by sentence index, not random phoneme instances
  - File: `src/phoneme_model.py` line ~325

- [x] **Inverse-frequency weighting**
  - Current: all phonemes weighted equally in the loss
  - Problem: rare phonemes (zh=16, oy=27 examples) have high-variance estimates
  - Fix: weighted ridge regression: `W = (X^T diag(w) X + αI)^{-1} X^T diag(w) Y` where w = 1/frequency
  - File: `src/phoneme_model.py:train_ridge()`

- [x] **Compute _TARGET_MAX from BEAT data**
  - Current: hand-tuned rescaling targets (jawOpen=0.55, etc.) from the original viseme map
  - Problem: arbitrary values tuned for one avatar, not empirically derived
  - Fix: compute 95th percentile per channel from BEAT frames, use as target max
  - File: downstream inference module's `_TARGET_MAX` dict

---

## Medium Effort (1-2 days each)

- [x] **Articulatory features instead of one-hot**
  - Current: one-hot phoneme encoding (124 dims), treats `b` and `p` as completely unrelated
  - Problem: can't learn that bilabial stops share lip closure; rare phonemes can't borrow from similar ones
  - Fix: replace with 6-8 dim articulatory features: place (bilabial/alveolar/velar), manner (stop/fricative/nasal), voicing, lip rounding, tongue height, nasality
  - Reduces features from 124 to ~25 dims, encodes similarity, helps rare phonemes
  - File: `src/phoneme_model.py:featurize()`

- [x] **Multiple frames per phoneme**
  - Current: extract only the midpoint frame per phoneme
  - Problem: for 40ms stop consonants, midpoint may still be in transition, not at apex
  - Fix: extract frames at 25%, 50%, 75% of phoneme duration, average for more robust targets
  - File: `src/data_format.py:get_all_labeled_frames()`

- [ ] **Small MLP (128→64→55)**
  - Current: ridge regression (linear model)
  - Problem: regression-to-mean is fundamental to linear models (CodeTalker CVPR 2023: "suffers from over-smoothed facial motions")
  - Fix: replace ridge with a 2-layer MLP with ReLU; captures non-linear coarticulation
  - Expected: 15-30% MAE reduction based on literature
  - File: `src/phoneme_model.py`

---

## Large Effort (days-weeks)

- [x] **Speaker normalization for BEAT**
  - Problem: 30 BEAT speakers have different resting faces, jaw sizes, lip thickness
  - Fix: per-speaker mean subtraction before training; predict delta from neutral, not absolute weights
  - References: VOCA (2019) uses one-hot speaker conditioning; FaceFormer (2022) uses learned embeddings

- [ ] **Sequence model (1D-CNN or LSTM)**
  - Problem: current model predicts per-phoneme independently, no temporal coherence
  - Fix: 1D-CNN with kernel size 5 (receptive field ~430ms) or bidirectional LSTM
  - Only if MLP still produces jerky transitions
  - References: VisemeNet (2018), FaceFormer (2022)

- [ ] **Generative model (VQ-VAE or diffusion)**
  - Problem: all regression models produce "average" faces, not crisp poses
  - Fix: discrete codebook (VQ-VAE) ensures only "real" mouth shapes are predicted
  - References: CodeTalker (2023), SAiD (2024)
  - This is the nuclear option — complete architecture change

---

## Known Data Issues

| Issue | Severity | Status |
|-------|----------|--------|
| Azure gold standard has compressed dynamic range | CRITICAL | Mitigated: BEAT model is now active |
| IPA→ARPAbet normalization for Inworld phonemes | FIXED | 43 Inworld symbols mapped to ARPAbet |
| BEAT 60→30 FPS downsampling loses fast consonant detail | MEDIUM | Known, could average frames instead |
| Azure viseme ID ambiguity (40→22 phonemes) | MEDIUM | Mitigated: CMUdict + BEAT bypass this |
| One-hot encoding treats all phonemes as equidistant | MEDIUM | FIXED: articulatory features in `src/articulatory.py` |
| Silence over-representation in training data | LOW | Rescaling handles the baseline |

---

## References

- VOCA (CVPR 2019): Speaker conditioning essential for multi-speaker training
- FaceFormer (CVPR 2022): Self-supervised speech features + periodic positional encoding
- CodeTalker (CVPR 2023): Regression-to-mean is fundamental; VQ-VAE codebook solves it
- BEAT/EMAGE (ECCV 2022 / CVPR 2024): 30-speaker dataset, compositional VQ-VAEs
- SAiD (2024): Diffusion on ARKit blendshapes, selected 32/52 channels as "crucial for speech"
- Bear & Harvey (2018): Optimal viseme set varies by speaker (11-35); standard mappings are suboptimal
