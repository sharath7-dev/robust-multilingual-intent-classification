# Robust and Calibrated Multilingual Intent Classification

A reproducible NLP project studying multilingual intent classification under spelling noise, transliterated input, unseen intents, and uncertainty-aware rejection.

## Current scope

Languages:

- English (`en-US`)
- Hindi (`hi-IN`)
- Kannada (`kn-IN`)

## Dataset

The project uses the public [MASSIVE 1.1 dataset](https://github.com/alexa/massive), licensed under CC BY 4.0. Dataset files are downloaded when required and are not stored in this repository.

Verified statistics:

- 49,563 examples across the selected languages
- 60 original intents
- 50 known training intents
- 5 validation-OOS intents
- 5 separate test-OOS intents
- Official dataset partitions preserved
- No held-out intent leakage into training

The frozen intent groups are stored in [`configs/intent_split.json`](configs/intent_split.json).

## Phase 1 verified baseline

Model: word-and-character TF-IDF with logistic regression.

| Evaluation group | Accuracy | Macro-F1 |
|---|---:|---:|
| Overall validation | 0.8369 | 0.8210 |
| English | 0.8594 | 0.8394 |
| Hindi | 0.8308 | 0.8140 |
| Kannada | 0.8205 | 0.8043 |

Training completed in 61.8 seconds. The solver converged after 115 iterations. The test set has not been evaluated.


## Phase 2 verified multilingual models

Two XLM-R-base models were fine-tuned for two epochs using the same 50 known intents.

| Model | Validation accuracy | Validation macro-F1 |
|---|---:|---:|
| TF-IDF baseline | 0.8369 | **0.8210** |
| Clean XLM-R | 0.8453 | 0.7625 |
| Weighted XLM-R | **0.8522** | 0.7667 |

Weighted XLM-R improved over clean XLM-R across all three languages:

| Language | Clean XLM-R macro-F1 | Weighted XLM-R macro-F1 |
|---|---:|---:|
| English | 0.7899 | **0.8027** |
| Hindi | 0.7559 | **0.7770** |
| Kannada | 0.7307 | **0.7443** |

TF-IDF remains the strongest clean-text model by macro-F1. Weighted XLM-R is retained for later robustness and transliteration comparisons. Large checkpoints are stored outside GitHub in Google Drive.

## Repository structure

- `configs/`: experiment configurations
- `docs/`: documentation
- `notebooks/`: Colab notebooks
- `results/metrics/`: evaluation metrics
- `results/figures/`: plots
- `src/`: reusable project code
- `ui/`: final interface

Datasets, secrets, environments, and model checkpoints are excluded from Git.

## Current status

- [x] Dataset verification
- [x] Leakage-safe OOS split
- [x] TF-IDF validation baseline
- [x] XLM-R multilingual model
- [ ] Noise robustness evaluation
- [ ] Transliteration diagnostic
- [ ] OOS detection and calibration
- [ ] Final test evaluation
- [ ] Professional interface

Only verified experimental results are reported.

<!-- SPELLING_NOISE_RESULTS_START -->
## Phase 3 — Synthetic spelling-noise robustness

We evaluated three models on the known-intent validation set under deterministic Unicode-aware spelling noise:

- TF-IDF with logistic regression
- Class-weighted XLM-R
- Noise-augmented, class-weighted XLM-R

The perturbations use single-grapheme deletion and adjacent-grapheme transposition at rates of 5%, 10%, and 15% per eligible word. All experiments use random seed 42.

### Noise augmentation

The augmented XLM-R was trained using:

- 26,193 clean training examples
- 5,967 synthetic noisy copies
- 1,989 noisy copies per language
- 32,160 total training examples
- Two training epochs
- No validation or test examples in training

### Overall validation results

| Model | Noise | Accuracy | Macro-F1 | Drop from clean |
|---|---:|---:|---:|---:|
| TF-IDF | 0% | 0.8369 | 0.8210 | 0.0000 |
| TF-IDF | 5% | 0.8283 | 0.8133 | 0.0076 |
| TF-IDF | 10% | 0.8177 | 0.7958 | 0.0252 |
| TF-IDF | 15% | 0.8062 | 0.7937 | 0.0273 |
| Weighted XLM-R | 0% | 0.8522 | 0.7667 | 0.0000 |
| Weighted XLM-R | 5% | 0.8365 | 0.7571 | 0.0096 |
| Weighted XLM-R | 10% | 0.8220 | 0.7383 | 0.0284 |
| Weighted XLM-R | 15% | 0.8075 | 0.7253 | 0.0413 |
| Noise-Augmented XLM-R | 0% | 0.8635 | 0.8221 | 0.0000 |
| Noise-Augmented XLM-R | 5% | 0.8516 | 0.8111 | 0.0110 |
| Noise-Augmented XLM-R | 10% | 0.8404 | 0.7817 | 0.0405 |
| Noise-Augmented XLM-R | 15% | 0.8280 | 0.7685 | 0.0536 |

Noise augmentation improves XLM-R's absolute accuracy and macro-F1 at every evaluated noise level. However, the augmented model still degrades relative to its own stronger clean score. TF-IDF retains the highest macro-F1 at 5%, 10%, and 15% noise, while the augmented XLM-R has the highest accuracy.

These experiments use controlled synthetic perturbations and do not establish performance on naturally occurring spelling errors. The final test set remains untouched.

![Spelling-noise robustness](results/figures/spelling_noise_robustness.png)
<!-- SPELLING_NOISE_RESULTS_END -->
