# ML Viseme Pipeline: Technical Learning Journal

A postmortem of building a phoneme-to-blendshape ML system for lip-sync animation. Documents the standard ML decision process, mistakes made at each stage, the concepts that explain them, and references for deeper study.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Choosing a Learning Approach](#2-choosing-a-learning-approach)
3. [Choosing Model Complexity](#3-choosing-model-complexity)
4. [Selecting a Gold Standard](#4-selecting-a-gold-standard)
5. [Data Collection and Bias](#5-data-collection-and-bias)
6. [Training and Evaluation](#6-training-and-evaluation)
7. [Deployment and Train-Serve Skew](#7-deployment-and-train-serve-skew)
8. [Regression to the Mean](#8-regression-to-the-mean)
9. [Summary of Mistakes](#9-summary-of-mistakes)
10. [Recommended Reading](#10-recommended-reading)

---

## 1. Problem Definition

### The Task

Convert phoneme labels with timing (from a TTS provider) into 55 ARKit blendshape weights at 30 FPS for 3D avatar lip-sync animation.

**Input:** `[{phone: "ah", start: 0.05s, duration: 0.1s}, ...]`
**Output:** `[{frame: 0, time: 0.0ms, weights: [55 floats]}, ...]`

### The Standard ML Process

Before writing any code, you should answer these questions:

1. **What type of learning?** (supervised / unsupervised / reinforcement)
2. **What model class?** (lookup table / linear / non-linear / generative)
3. **What is the gold standard?** (where do correct outputs come from)
4. **What are the evaluation metrics?** (mathematical + perceptual)
5. **What are the deployment constraints?** (latency, memory, fallback)

We answered some well and some poorly. Each section below covers one of these decisions.

---

## 2. Choosing a Learning Approach

### The Decision Framework

| Question | Answer | Implies |
|----------|--------|---------|
| Do you have labeled (input, output) pairs? | Yes — phonemes paired with blendshape frames | **Supervised learning** |
| Is the output continuous or discrete? | Continuous (55 floats in [0,1]) | **Regression**, not classification |
| Does temporal order matter? | Partially — coarticulation spans 2-3 phonemes | Sequence features help but aren't required |
| Is there a clear "correct" answer? | Yes — the gold standard's blendshape at each phoneme | Deterministic regression works |

**Our choice:** Supervised regression. Correct for this task.

**When other approaches would be better:**
- **Unsupervised** — if you don't have gold standard blendshapes and need to discover mouth shape clusters from unlabeled data
- **Reinforcement** — if quality is defined by viewer preference feedback (no single "correct" answer)
- **Generative** — if the output should be diverse rather than deterministic (e.g., different valid pronunciations of the same phoneme). This is what CodeTalker and SAiD use — see [Section 8](#8-regression-to-the-mean)

### References

- **Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*, Chapter 1.** The standard introduction to supervised vs unsupervised learning, when to use each, and how to formalize a learning problem. Free PDF available at Microsoft Research.
- **Stanford CS229 Lecture 1 (Andrew Ng).** Covers the supervised learning setup, hypothesis classes, and loss functions in a concrete, accessible way. Available on YouTube.

---

## 3. Choosing Model Complexity

### The Bias-Variance Tradeoff

| Model | Parameters | What It Can Learn | Risk |
|-------|-----------|-------------------|------|
| Lookup table (10 categories × 55 weights) | 550 | One static pose per category | **High bias** — can't distinguish "s" from "k" in same category |
| Ridge regression (40 phonemes × 55 weights) | 6,875 | Per-phoneme pose conditioned on context | **Low bias, moderate variance** |
| MLP (128→64→55) | ~25,000 | Non-linear phoneme interactions | **Low bias, higher variance** |
| Transformer/LSTM | 100K+ | Full temporal dynamics | **Lowest bias, highest variance** — needs much more data |

**The rule:** start with the simplest model that has enough capacity, then add complexity only when you have evidence the simple model is the bottleneck.

### What We Did Right

Started with the lookup table, identified its limitations through PCA (PC1 at 43-49% — high intra-category variance), then upgraded to ridge regression. This is the correct incremental approach.

### What We Did Wrong

**Skipped measuring baseline performance.** We never measured what the hand-tuned lookup table actually scored on the evaluation metrics before building the ML system. This meant we couldn't tell if the ML model was actually better or just different.

**Rule:** Always measure your baseline first. If the simple approach scores 0.90 and the complex approach scores 0.91, the complexity isn't worth it.

### The Concept: How to Know When to Upgrade

1. **Compute validation error** on the current model
2. **Check train/val gap:** if train error >> val error, you're underfitting → more capacity needed. If train error << val error, you're overfitting → more regularization or data needed.
3. **Check residuals:** are the errors systematic (same phonemes always wrong) or random? Systematic errors suggest missing features or model limitations.

Our train/val gap was +0.0004 (good — no overfitting). But the residuals were systematic: all consonant channels had high error, all vowel channels were fine. This points to a **feature representation problem** (one-hot encoding treats similar phonemes as unrelated), not a model capacity problem.

### References

- **Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, Chapter 7: Model Assessment and Selection.** The definitive treatment of bias-variance tradeoff, cross-validation, and model selection. Free PDF at Stanford.
- **Google. "Rules of Machine Learning: Best Practices for ML Engineering."** Practical rules like "Don't be afraid to launch a model without machine learning" and "Measure first, then optimize." Covers the incremental approach to model complexity. Available at developers.google.com.

---

## 4. Selecting a Gold Standard

### The Critical Error

We used Azure TTS's neural blendshape output as the gold standard — the "correct" answer to train against. This is a **neural model's prediction**, not ground truth. Azure's DragonHDOmni was designed for its own rendering pipeline, not for driving external ARKit avatars. Its output has:

- **Compressed dynamic range** — jawOpen never drops below 0.15 or exceeds 0.70
- **Absent bilabial closure** — "b"/"p"/"m" don't show proper lip pressing
- **Grouped phonemes** — 40 phonemes collapsed into 22 viseme IDs

We achieved 0.955 quality against Azure and the result looked terrible on the avatar. **The model was 95.5% accurate at being wrong.**

### The Concept: Goodhart's Law in ML

> "When a measure becomes a target, it ceases to be a good measure."

If your benchmark doesn't reflect real-world quality, optimizing against it pushes you away from real-world quality. This happens when:

1. **Your benchmark is a proxy, not the true objective.** MAE against Azure's output is a proxy for "looks good on the avatar." These can diverge.
2. **Your benchmark has systematic biases.** Azure's compressed range is a systematic bias that the model faithfully reproduces.
3. **You never validate against the true objective.** We didn't render on the avatar until after weeks of development.

### How to Select a Good Benchmark

| Criterion | Azure TTS | BEAT (iPhone capture) | Animator judgment |
|-----------|-----------|----------------------|-------------------|
| Represents real mouth shapes? | No (neural prediction) | Yes (real human) | Yes (visual quality) |
| Covers all phonemes? | 22 groups | 40 individual | N/A |
| Multiple speakers? | 1 voice | 30 speakers | N/A |
| Quantifiable? | Yes (MAE) | Yes (MAE) | No (subjective) |
| Scalable? | Yes (API) | Yes (pre-recorded) | No (manual) |

**The right approach:** use a quantifiable benchmark (BEAT) for development, but validate periodically against the true objective (animator rendering). The quantifiable metric tells you "am I improving?" but only the true objective tells you "is it good enough?"

### References

- **Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems." NIPS.** Describes how ML systems accumulate hidden dependencies, including on flawed benchmarks. The foundational paper on ML system design. Available at papers.nips.cc.
- **Krakovna, V. (2020). "Specification Gaming: The Flip Side of AI Ingenuity." DeepMind Blog.** Examples of systems that optimize metrics while violating the intended objective. Directly relevant to Goodhart's Law in ML.
- **Dehghani, M., et al. (2021). "The Benchmark Lottery." NeurIPS.** Shows how benchmark choice biases ML research — models that win on one benchmark may lose on another. Argues for multi-dimensional evaluation.

---

## 5. Data Collection and Bias

### Biases We Found

| Bias | Source | Impact | Detection Method |
|------|--------|--------|------------------|
| **Single speaker** | Azure uses one voice (Ava) | Model learns one person's articulation | Check speaker count |
| **Phoneme ambiguity** | Azure groups 40→22 phonemes | "b"/"p"/"m" indistinguishable | Compare viseme ID mappings to ARPAbet |
| **Domain shift** | Harvard sentences (formal) vs production (conversational) | Model trained on clear enunciation, deployed on casual speech | Compare phoneme distributions |
| **Silence over-representation** | Azure baseline face ≠ closed mouth | Silence weights are non-zero average, not rest pose | Check frame label distribution |
| **One-hot feature blindness** | `b` and `p` encoded as orthogonal | Can't learn that bilabials share lip closure | Analyze feature representation |
| **Midpoint frame sampling** | Extract only phoneme midpoint | Misses transition dynamics, short consonants may be at wrong frame | Check frame selection against phoneme duration |

### How to Detect Bias Before It Hurts

1. **Distribution analysis:** Plot phoneme frequencies. If "zh" has 16 examples and "t" has 3,524, your model's "zh" predictions are unreliable. Use inverse-frequency weighting.

2. **PCA per class:** If PC1 < 60% within a category, the category is too heterogeneous. Split it or use finer labels.

3. **Residual analysis:** After training, check which phonemes have the highest error. If all bilabials are wrong, the problem is systematic (feature representation), not random (insufficient data).

4. **Domain comparison:** Compare your training data distribution to your production data distribution. If the distributions differ (formal vs casual, one speaker vs many), expect degraded performance.

### The Concept: Train/Test Split Strategy

| Strategy | When to Use | Pitfall |
|----------|-------------|---------|
| Random split of examples | Independent examples | Leaks temporal correlation if examples come from the same sequence |
| Split by sequence/sentence | Time-series or sequential data | Phonemes from the same sentence share prosody |
| Split by speaker | Multi-speaker data | Must test generalization to unseen speakers |
| Stratified by class | Imbalanced classes | Ensures rare classes appear in both train and test |

**Our mistake:** We split individual phoneme instances randomly (85/15). Phonemes from the same sentence share temporal context, creating data leakage. The correct approach for this task is **sentence-level split** (all phonemes from a sentence go to either train or val, never both).

### References

- **Moreno-Torres, J.G., et al. (2012). "A Unifying View on Dataset Shift in Classification." Pattern Recognition.** Comprehensive taxonomy of distribution shift types (covariate shift, prior probability shift, concept shift) with detection methods.
- **Stanford CS329S: Machine Learning Systems Design.** Covers data management, distribution shift, and data quality in production ML. Lecture notes available online.
- **Raschka, S. (2018). "Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning."** Deep dive on cross-validation strategies, including nested CV, bootstrap, and temporal splits. arXiv:1811.12808.

---

## 6. Training and Evaluation

### Evaluation Metrics We Used

| Metric | What It Measures | Limitation |
|--------|------------------|------------|
| MAE (per channel) | Average absolute error on each blendshape | Treats all errors equally — a 0.05 error on jawOpen matters more than on mouthDimple |
| RMSE | Penalizes large errors more than MAE | Still treats all channels equally |
| Pearson correlation | Temporal pattern match (scale-independent) | A sequence with the right pattern but wrong magnitude scores well |
| Velocity MAE | Transition speed correctness | Captures dynamics but sensitive to timing alignment |
| Perceptual-weighted MAE | Weights channels by visual importance | The weights themselves are subjective |
| Viseme hit rate | Binary "does the shape reach 80% of target?" | Threshold is arbitrary |

### What's Missing: Perceptual Evaluation

All our metrics are mathematical. None measure "does it look right to a human?" In the speech/animation field, standard perceptual evaluation methods include:

1. **MUSHRA test:** Multiple Stimuli with Hidden Reference and Anchor. Raters score animation quality on a 0-100 scale, with a hidden reference and known bad example.
2. **A/B preference test:** Show two animations side by side, ask "which looks more natural?"
3. **MOS (Mean Opinion Score):** Raters score each animation independently on a 1-5 scale.

We skipped all of these. Our "perceptual evaluation" was the animator looking at the export and saying "the mouth is stuck open." This is valuable but unsystematic.

### The Concept: Mathematical Metrics ≠ Perceptual Quality

A model with MAE 0.04 on jawOpen might look better or worse than one with MAE 0.06 — it depends on whether the errors correlate with visually salient moments. A jawOpen error during a vowel is very visible; the same error during a silence is invisible.

**Rule of thumb:** mathematical metrics are useful for comparing iterations during development (is version N+1 better than N?). But the absolute value of the metric doesn't tell you if the model is "good enough" — only perceptual evaluation answers that.

### References

- **ITU-R BS.1534-3 (MUSHRA).** The international standard for subjective quality evaluation of audio codecs. Adapted for speech animation quality in recent papers.
- **Haque, A., et al. (2025). "The Wild West of Evaluating Speech-Driven 3D Facial Animation." Eurographics.** Comprehensive review of evaluation practices in speech-driven animation. Found that most papers lack perceptual evaluation and that mathematical metrics often don't correlate with perceived quality.
- **Raschka, S. (2018). "Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning."** arXiv:1811.12808. Covers the theory behind train/test splitting, cross-validation, and when simple metrics are insufficient.

---

## 7. Deployment and Train-Serve Skew

### The IPA→ARPAbet Bug

The model was trained on ARPAbet phonemes (from Azure/BEAT): `ah`, `t`, `sh`, `ng`.
Production TTS (Inworld) sends IPA phonemes: `ə`, `t`, `ʃ`, `ŋ`.

37 out of 43 Inworld phonemes were unrecognized. The model fell back to its bias term for every unknown phoneme — producing identical output regardless of what was being said.

This is a textbook **train-serve skew**: the training pipeline and the inference pipeline process inputs differently.

### How to Prevent Train-Serve Skew

Google's ML Test Score framework (Breck et al., 2017) proposes these checks:

1. **Feature parity test:** Verify that every feature computed during training can be computed identically during inference. In our case, the phoneme vocabulary was different.

2. **Distribution test:** Compare the distribution of features at training time vs inference time. If inference sees values outside the training distribution, flag it.

3. **Integration test:** Run a small set of production inputs through the full inference pipeline and check that the output is reasonable. We didn't do this until animator export testing.

### The Fix Pattern

Add a **normalization layer** between raw input and the model. In our case, an IPA→ARPAbet translation table. This layer guarantees that the model always sees the vocabulary it was trained on, regardless of what the provider sends.

This is better than retraining on IPA because:
- It works for any future provider without retraining
- It's stateless and adds zero latency (dict lookup)
- It fails visibly (unknown symbol → "sil" fallback, easy to log and detect)

### References

- **Breck, E., et al. (2017). "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction." IEEE Big Data.** A checklist of 28 tests for ML systems covering data, model, infrastructure, and monitoring. The standard reference for ML deployment quality.
- **Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems." NIPS.** Describes how ML systems accumulate hidden dependencies that create deployment failures. The "normalization layer" pattern is an example of an explicitly managed dependency.

---

## 8. Regression to the Mean

### The Fundamental Problem with Ridge Regression

Ridge regression minimizes `||Y_pred - Y_true||² + α||W||²`. For any input, it predicts the **conditional mean** of all matching training examples. If the training data shows five different mouth shapes for the phoneme "ah" in the same context, ridge regression predicts their average — which may not look like any real mouth shape.

This is why the model produces "mushy" outputs: every prediction is an average, never a crisp pose.

### How Other Approaches Solve This

| Approach | How It Avoids Averaging | Tradeoff |
|----------|------------------------|----------|
| **MLP + ReLU** | Non-linear decision boundaries can learn sharper conditional means | Still deterministic; better but not crisp |
| **VQ-VAE** (CodeTalker) | Quantizes output to a learned codebook of "real" poses — prediction is always a valid pose from the codebook | Codebook must cover all valid poses; training is more complex |
| **Diffusion** (SAiD) | Generates diverse samples from the learned distribution — different outputs for the same input | Stochastic; may produce different results on each run |
| **GAN** | Adversarial loss ensures outputs are indistinguishable from real data | Training instability; mode collapse risk |

**Our model** uses ridge regression (deterministic linear). The output is always the average, never a specific real pose. This is the simplest approach and the accuracy ceiling is inherently limited.

### When to Upgrade

The decision to move from linear regression to a more complex model should be driven by evidence, not speculation:

1. **Measure residual structure:** If errors are systematically correlated (e.g., all consonants wrong), a non-linear model might capture the pattern.
2. **Check perceptual quality:** If the mathematical metrics are good but the animation looks "floaty" or "average," you've hit the regression-to-mean ceiling.
3. **Try a small MLP first:** A 2-layer MLP (128→64→55) with ReLU adds non-linearity with minimal complexity. If it significantly outperforms ridge, pursue further. If not, the problem is elsewhere (data quality, feature representation).

### References

- **Mathieu, M., Couprie, C., & LeCun, Y. (2016). "Deep Multi-Scale Video Prediction Beyond Mean Square Error." ICLR.** The foundational paper showing why L2 loss produces blurry/average predictions and how adversarial and gradient difference losses produce sharper outputs.
- **Xing, J., et al. (2023). "CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior." CVPR.** Directly addresses regression-to-mean in speech animation. Uses VQ-VAE codebook to ensure only "real" poses are predicted.
- **Kim, J., et al. (2024). "SAiD: Speech-Driven Blendshape Facial Animation with Diffusion." arXiv:2401.08655.** Applies denoising diffusion to blendshape prediction. Outputs diverse, natural poses instead of averages.
- **BEAT / EMAGE. Liu, H., et al. (2024). "EMAGE: Towards Unified Holistic Co-Speech Gesture Generation via Expressive Masked Audio Gesture Modeling." CVPR.** Uses compositional VQ-VAEs for multi-part body+face generation from speech.

---

## 9. Summary of Mistakes

| Phase | Mistake | Root Cause | What We Should Have Done |
|-------|---------|------------|--------------------------|
| 1 | Added 720 sentences to a 10-entry lookup table | Didn't match model capacity to data complexity | Run PCA first to check if categories are valid |
| 3 | Optimized MAE against Azure, got 0.955 quality with unusable output | Treated a proxy metric as the true objective (Goodhart's Law) | Render on avatar before celebrating metrics |
| 4 | Added dynamic range rescaling as a post-processing band-aid | Fixed the symptom (compressed range) not the cause (Azure's biased data) | Switch to BEAT (real capture) data from the start |
| 5 | CMUdict disambiguation improved labels but not targets | Correct inputs + wrong targets = wrong model | Use a gold standard with correct targets first |
| 6 | Tried to set up MFA for BEAT when BEAT already had phoneme timing | Didn't inspect the data before building preprocessing | Read a few data files before writing any code |
| 7 | 37/43 Inworld phonemes unrecognized at inference time | Never tested with production data | Add integration test with real provider input early |
| 8 | Used temperature for Inworld emotion (sampling randomness ≠ emotion) | Assumed all providers use the same abstraction | Read the provider's API docs for the features you're using |

---

## 10. Recommended Reading

### If You Have 2 Hours

1. **Google. "Rules of Machine Learning."** (30 min) — Practical rules for ML engineering. Start here.
   - developers.google.com/machine-learning/guides/rules-of-ml
2. **Sculley et al. (2015). "Hidden Technical Debt in ML Systems."** (1 hr) — Why ML systems are harder to maintain than regular software.
   - papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html

### If You Have a Weekend

3. **Hastie, Tibshirani & Friedman. *Elements of Statistical Learning*, Chapter 7.** (3 hr) — Bias-variance tradeoff, cross-validation, model selection. The mathematical foundation.
   - Free PDF: hastie.su.domains/ElemStatLearn/
4. **Breck et al. (2017). "ML Test Score."** (1 hr) — Checklist for ML production readiness.
   - IEEE Big Data 2017
5. **Mathieu et al. (2016). "Beyond Mean Square Error."** (1 hr) — Why L2 loss produces blurry outputs.

### For Speech-Driven Animation Specifically

6. **CodeTalker (CVPR 2023)** — Discrete motion priors for crisp facial animation
   - arxiv.org/abs/2301.02379
7. **SAiD (2024)** — Diffusion for ARKit blendshape generation
   - arxiv.org/abs/2401.08655
8. **FaceFormer (CVPR 2022)** — Transformer-based speech-to-face with self-supervised audio features
   - arxiv.org/abs/2112.05329
9. **BEAT/EMAGE (ECCV 2022 / CVPR 2024)** — 76h multi-speaker gesture+face dataset
   - pantomatrix.github.io/BEAT/
10. **Haque et al. (2025). "Wild West of Evaluating Speech-Driven 3D Facial Animation."** — Why current evaluation practices are insufficient
    - Eurographics 2025

### For ML Systems Engineering

11. **Stanford CS329S: ML Systems Design** — Data management, distribution shift, monitoring
    - stanford-cs329s.github.io
12. **Moreno-Torres et al. (2012). "Dataset Shift in Classification."** — Taxonomy of distribution shift types
    - Pattern Recognition 45(1)
13. **Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*.** — The full ML textbook, covers everything from probability to neural nets
    - Free PDF at Microsoft Research
