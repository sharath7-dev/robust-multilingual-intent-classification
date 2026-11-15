"""Download and verify the frozen model during a Vercel build."""

from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "artifacts" / "models" / "xlmr_combined_augmented_final"
REQUIRED_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
}


def main() -> None:
    repo_id = os.environ.get("MODEL_REPO_ID", "").strip()
    revision = os.environ.get("MODEL_REVISION", "main").strip() or "main"
    token = os.environ.get("HF_TOKEN") or None

    if not repo_id:
        raise RuntimeError(
            "MODEL_REPO_ID is required. Point it to the Hugging Face model "
            "repository containing xlmr_combined_augmented_final."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=MODEL_DIR,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        ],
    )

    present = {path.name for path in MODEL_DIR.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_FILES - present)
    if missing:
        raise RuntimeError(f"Downloaded model is incomplete; missing: {missing}")

    weights = MODEL_DIR / "model.safetensors"
    if weights.stat().st_size < 500_000_000:
        raise RuntimeError(
            "model.safetensors is unexpectedly small; refusing to deploy an "
            "incomplete checkpoint."
        )

    info = HfApi(token=token).model_info(repo_id=repo_id, revision=revision)
    metadata = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_commit": info.sha,
        "weight_bytes": weights.stat().st_size,
    }
    (MODEL_DIR / "deployment_model.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        "Deployment model verified:",
        repo_id,
        info.sha,
        f"{weights.stat().st_size / (1024 ** 3):.2f} GiB",
    )


if __name__ == "__main__":
    main()
