# Permanent Vercel Deployment

IntentShield can run as one Vercel Python Function. The website and the frozen
Combined-Augmented XLM-R classifier are deployed together, so the permanent URL
does not depend on a running Google Colab session.

## Architecture

1. GitHub stores the application, configurations, and verified results.
2. A Hugging Face model repository stores the 1.05 GB frozen checkpoint.
3. Vercel downloads that exact checkpoint during its build.
4. Vercel serves the mounted Gradio interface through a FastAPI ASGI entrypoint.

The deployment preserves the frozen temperature and acceptance threshold from
`configs/final_test_protocol.json`. It does not retrain or retune the model.

## One-time model upload

Create a Hugging Face **model** repository. Public is simplest; a private model
repository requires an `HF_TOKEN` secret in Vercel. Upload the contents of:

`xlmr_combined_augmented_final/`

The repository root must contain at least:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

After upload, copy the model repository ID and the commit hash shown in its
history. The commit hash pins deployment to the exact frozen checkpoint.

## Vercel project settings

Import the GitHub repository into Vercel, then add these variables to Preview
and Production before deploying:

| Variable | Value |
|---|---|
| `MODEL_REPO_ID` | `your-hf-name/your-model-repo` |
| `MODEL_REVISION` | immutable Hugging Face commit hash |
| `VERCEL_SUPPORT_LARGE_FUNCTIONS` | `1` |
| `HF_TOKEN` | only for a private model repository |

Keep Fluid Compute enabled. The repository pins Python 3.13 and places the
Function in Mumbai (`bom1`). The first request after scale-to-zero can be slower
because the 1.05 GB model must be loaded into memory, but the site wakes
automatically and never needs a Colab runtime.

## Deployment verification

Check all of the following before sharing the URL:

1. `/healthz` returns `status: ok`.
2. English, Hindi, Kannada, typo, and Romanized examples produce results.
3. A low-confidence unsupported request pauses instead of executing an action.
4. Browser microphone permission and spoken reply work in Chrome or Edge.
5. The Vercel Function logs contain no memory or timeout failures.

The Hobby plan provides 2 GB RAM and may have noticeable cold starts for this
model. If deployed inference exceeds memory, use a Vercel Pro Performance
Function or deploy the model behind a dedicated inference service; do not swap
in a different model without evaluating and documenting the change.
