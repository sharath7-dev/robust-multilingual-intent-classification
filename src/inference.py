"""Calibrated inference for the frozen combined XLM-R intent classifier."""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]


class IntentClassifier:
    def __init__(self, model_dir=None, device=None):
        configured_path = model_dir or os.environ.get("MODEL_DIR")
        self.model_dir = Path(configured_path) if configured_path else (
            REPO_ROOT / "artifacts/models/xlmr_combined_augmented_final"
        )
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                "Combined model directory not found. Set MODEL_DIR to "
                "xlmr_combined_augmented_final."
            )

        protocol = json.loads(
            (REPO_ROOT / "configs/final_test_protocol.json").read_text(encoding="utf-8")
        )
        mapping = json.loads(
            (REPO_ROOT / "configs/label_mapping.json").read_text(encoding="utf-8")
        )

        self.model_name = protocol["primary_final_model"]
        self.temperature = float(protocol["frozen_temperatures"][self.model_name])
        self.threshold = float(protocol["frozen_confidence_thresholds"][self.model_name])
        self.max_length = 64
        self.labels = {int(key): value for key, value in mapping["id2label"].items()}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, local_files_only=True
        ).to(self.device)
        self.model.eval()

        model_labels = {int(key): value for key, value in self.model.config.id2label.items()}
        if model_labels != self.labels:
            raise ValueError("Saved model labels do not match repository labels.")

    @torch.inference_mode()
    def predict(self, text, top_k=5):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a non-empty request.")
        if len(text) > 10_000:
            raise ValueError("Please keep the request below 10,000 characters.")

        full_tokens = self.tokenizer(text, truncation=False, return_attention_mask=False)[
            "input_ids"
        ]
        inputs = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**inputs).logits.float()
        probabilities = torch.softmax(logits / self.temperature, dim=-1)[0]
        values, indices = probabilities.topk(top_k)

        confidence = float(values[0])
        candidate = self.labels[int(indices[0])]
        accepted = confidence >= self.threshold
        return {
            "input": text,
            "status": "accepted" if accepted else "uncertain",
            "intent": candidate if accepted else None,
            "top_candidate": candidate,
            "confidence": confidence,
            "threshold": self.threshold,
            "temperature": self.temperature,
            "truncated": len(full_tokens) > self.max_length,
            "token_count": len(full_tokens),
            "top_predictions": [
                {"intent": self.labels[int(index)], "probability": float(value)}
                for value, index in zip(values, indices)
            ],
            "model": self.model_name,
        }
