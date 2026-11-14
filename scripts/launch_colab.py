"""Launch the IntentShield demo reliably from an already cloned Colab repo."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVE_MODEL = Path(
    "/content/drive/MyDrive/robust_multilingual_intent/"
    "artifacts/models/xlmr_combined_augmented_final"
)
LOCAL_MODEL = Path("/content/intent_shield_model")


def main() -> None:
    source_weights = DRIVE_MODEL / "model.safetensors"
    if not source_weights.is_file():
        raise FileNotFoundError(
            "Google Drive model was not found. Mount Drive first and verify: "
            f"{DRIVE_MODEL}"
        )

    target_weights = LOCAL_MODEL / "model.safetensors"
    needs_copy = (
        not target_weights.is_file()
        or target_weights.stat().st_size != source_weights.stat().st_size
    )
    if needs_copy:
        print("Copying the trained model from Drive to fast Colab storage...", flush=True)
        shutil.copytree(DRIVE_MODEL, LOCAL_MODEL, dirs_exist_ok=True)
    else:
        print("Using the existing local model cache.", flush=True)

    os.environ["MODEL_DIR"] = str(LOCAL_MODEL)
    os.environ["GRADIO_SHARE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"

    print("Starting IntentShield. The public gradio.live URL appears after model loading.", flush=True)
    os.chdir(REPO_ROOT)
    os.execv(sys.executable, [sys.executable, "-u", str(REPO_ROOT / "ui/app.py")])


if __name__ == "__main__":
    main()
