from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
RESULT_MAP = {
    "contradiction": "Contradicted",
    "entailment": "Supported",
    "neutral": "Insufficient evidence",
}


def _resolve_label_order(model: Any) -> list[str]:
    raw_mapping = getattr(model.config, "id2label", None) or {}
    resolved: dict[int, str] = {}

    for key, label in raw_mapping.items():
        try:
            resolved[int(key)] = str(label).lower()
        except (TypeError, ValueError):
            continue

    if len(resolved) != 3:
        resolved = {0: "contradiction", 1: "entailment", 2: "neutral"}

    ordered = [resolved.get(idx, "") for idx in range(3)]
    if any(label not in set(RESULT_MAP) for label in ordered):
        raise ValueError(
            f"Unexpected label mapping loaded for {MODEL_NAME}: {ordered}. "
            "Expected contradiction, entailment, neutral."
        )

    return ordered


@lru_cache(maxsize=1)
def load_nli_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to("cpu")
    model.eval()
    return tokenizer, model


def validate_inputs(source: str, claim: str, tokenizer: Any, model: Any) -> None:
    if source is None or claim is None:
        raise ValueError("Both the source excerpt and the claim are required.")

    source_clean = source.strip()
    claim_clean = claim.strip()
    if not source_clean or not claim_clean:
        raise ValueError("Please enter both a source excerpt and a claim before checking.")

    max_position_embeddings = getattr(model.config, "max_position_embeddings", None)
    if max_position_embeddings is None:
        max_position_embeddings = getattr(tokenizer, "model_max_length", None)
    if max_position_embeddings is None or max_position_embeddings <= 0:
        return

    token_count = len(
        tokenizer(
            source_clean,
            claim_clean,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
    )

    if token_count > max_position_embeddings:
        raise ValueError(
            "The combined source and claim is too long for the model's token limit "
            f"({max_position_embeddings} tokens including special tokens). Please shorten "
            "the excerpt or claim and try again."
        )


def run_nli_check(source: str, claim: str, tokenizer: Any | None = None, model: Any | None = None) -> dict[str, Any]:
    if tokenizer is None or model is None:
        tokenizer, model = load_nli_model()

    validate_inputs(source, claim, tokenizer, model)

    model_inputs = tokenizer(
        source,
        claim,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    with torch.inference_mode():
        logits = model(**model_inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)

    label_order = _resolve_label_order(model)
    scores = {label: float(probabilities[idx].item()) for idx, label in enumerate(label_order)}
    predicted_index = int(torch.argmax(probabilities).item())
    predicted_label = label_order[predicted_index]

    return {
        "result": RESULT_MAP[predicted_label],
        "predicted_label": predicted_label,
        "scores": scores,
        "label_order": label_order,
    }
