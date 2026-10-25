# Model Artifacts

Large trained models are not stored in this GitHub repository.

## Verified artifacts

The artifacts were produced in Google Colab and verified under:

`MyDrive/robust_multilingual_intent/artifacts/models/`

| Artifact | Type | Approximate size | Purpose |
|---|---|---:|---|
| `tfidf_baseline.joblib` | Joblib file | 27.56 MB | TF-IDF baseline |
| `xlmr_clean_final/` | Transformers model | 1.05 GB | Clean XLM-R baseline |
| `xlmr_weighted_final/` | Transformers model | 1.05 GB | Class-weighted XLM-R |
| `xlmr_noise_augmented_final/` | Transformers model | 1.05 GB | Noise-augmented XLM-R |
| `xlmr_combined_augmented_final/` | Transformers model | 1.05 GB | Final combined model |

The primary classifier uses `xlmr_combined_augmented_final/`.
The TF-IDF model is retained as a comparison baseline and achieved the
strongest frozen-test unseen-intent AUROC and OOS F1.

## Expected local structure

Place the required artifacts outside Git tracking:

    artifacts/
    └── models/
        ├── tfidf_baseline.joblib
        └── xlmr_combined_augmented_final/

The application will also accept an external combined-model path through
the `MODEL_DIR` environment variable.

## Reproduction and availability

The artifacts currently exist in the project owner's private Google Drive.
They are not claimed to be publicly downloadable from this repository.

They can be reproduced using:

- `notebooks/multilingual_intent_robustness_experiments.ipynb`
- the frozen configurations in `configs/`
- the dependency versions in `requirements.txt`

Dataset files and trained checkpoints must not be committed to GitHub.
