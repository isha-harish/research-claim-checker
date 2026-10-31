from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Any

import numpy as np
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from checker import load_nli_model, run_nli_check

MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 30
MAX_CLAIMS = 5
MAX_PASSAGES = 500
RETRIEVAL_TOP_K = 3
EMBEDDING_TARGET_TOKENS = 180
NLI_TOKEN_LIMIT = 512
OVERLAP_SENTENCES = 1


@dataclass
class Passage:
    passage_id: str
    page_number: int
    text: str


def _normalize_text(value: str) -> str:
    value = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


@lru_cache(maxsize=1)
def _get_embedding_tokenizer() -> Any | None:
    try:
        return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


def _estimate_tokens(text: str) -> int:
    tokenizer = _get_embedding_tokenizer()
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=True))
        except Exception:
            pass
    return max(1, len(re.findall(r"\S+", text)))


def _split_into_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    if not parts:
        return [text.strip()] if text.strip() else []
    return parts


def _pack_sentences(sentences: list[str], target_tokens: int = EMBEDDING_TARGET_TOKENS, overlap_count: int = OVERLAP_SENTENCES) -> list[str]:
    if not sentences:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for sentence in sentences:
        clean = re.sub(r"\s+", " ", sentence).strip()
        if not clean:
            continue
        sentence_tokens = _estimate_tokens(clean)
        if sentence_tokens > target_tokens:
            if buffer:
                chunks.append(" ".join(buffer))
                overlap = buffer[-overlap_count:] if overlap_count > 0 else []
                buffer = overlap[:] if overlap else []
                buffer_tokens = sum(_estimate_tokens(part) for part in buffer)
            words = clean.split()
            while words:
                group: list[str] = []
                group_tokens = 0
                while words and group_tokens + _estimate_tokens(words[0]) <= target_tokens:
                    word = words.pop(0)
                    group.append(word)
                    group_tokens = _estimate_tokens(" ".join(group))
                if group:
                    chunks.append(" ".join(group))
                if not words:
                    break
            continue

        candidate = " ".join(buffer + [clean]) if buffer else clean
        candidate_tokens = _estimate_tokens(candidate)
        if candidate_tokens <= target_tokens:
            buffer.append(clean)
            buffer_tokens = candidate_tokens
        else:
            if buffer:
                chunks.append(" ".join(buffer))
                overlap = buffer[-overlap_count:] if overlap_count > 0 else []
                buffer = overlap[:]
                buffer_tokens = sum(_estimate_tokens(part) for part in buffer)
            if not buffer:
                buffer = [clean]
                buffer_tokens = sentence_tokens
            else:
                candidate_with_overlap = " ".join(buffer + [clean])
                if _estimate_tokens(candidate_with_overlap) <= target_tokens:
                    buffer.append(clean)
                    buffer_tokens = _estimate_tokens(" ".join(buffer))
                else:
                    chunks.append(" ".join(buffer))
                    buffer = [clean]
                    buffer_tokens = sentence_tokens

    if buffer:
        chunks.append(" ".join(buffer))

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _segment_page_text(page_text: str) -> list[str]:
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n+", page_text) if p.strip()]
    if not paragraphs:
        paragraphs = [re.sub(r"\s+", " ", page_text).strip()]

    parts: list[str] = []
    for paragraph in paragraphs:
        sentences = _split_into_sentences(paragraph)
        if not sentences:
            parts.append(paragraph)
            continue
        parts.extend(_pack_sentences(sentences, target_tokens=EMBEDDING_TARGET_TOKENS, overlap_count=OVERLAP_SENTENCES))

    return parts


def extract_pdf_text(uploaded_file) -> tuple[list[dict[str, Any]], int, int, int, str | None]:
    if uploaded_file is None:
        raise ValueError("Please upload a PDF before analyzing claims.")

    contents = uploaded_file.getvalue()
    if not contents:
        raise ValueError("The uploaded PDF is empty.")
    if len(contents) > MAX_PDF_SIZE_BYTES:
        raise ValueError(f"The uploaded PDF is larger than {MAX_PDF_SIZE_BYTES // (1024 * 1024)} MB. Please upload a smaller file.")

    try:
        reader = PdfReader(BytesIO(contents))
    except (PdfReadError, ValueError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable PDF.") from exc

    if reader.is_encrypted:
        raise ValueError("This PDF is encrypted and cannot be processed without a password.")

    total_pages = len(reader.pages)
    if total_pages > MAX_PDF_PAGES:
        raise ValueError(f"This PDF has more than {MAX_PDF_PAGES} pages. Please upload a smaller file.")

    pages_with_text: list[dict[str, Any]] = []
    empty_pages = 0

    for page_index in range(total_pages):
        page = reader.pages[page_index]
        page_number = page_index + 1
        raw_text = page.extract_text() or ""
        cleaned_text = _normalize_text(raw_text)
        if not cleaned_text:
            empty_pages += 1
            continue
        pages_with_text.append({"page_number": page_number, "text": cleaned_text})

    if not pages_with_text:
        raise ValueError("No usable text could be extracted from this PDF. Text-based PDFs only are supported.")

    partial_warning = None
    if empty_pages:
        partial_warning = f"{empty_pages} page(s) contained no extractable text; those pages were skipped. Text order may also be imperfect in some PDFs."

    return pages_with_text, total_pages, len(pages_with_text), empty_pages, partial_warning


def build_passages(pages_with_text: list[dict[str, Any]]) -> list[Passage]:
    passages: list[Passage] = []
    passage_index = 0
    for page in pages_with_text:
        page_number = page["page_number"]
        for chunk in _segment_page_text(page["text"]):
            passage_index += 1
            passages.append(Passage(passage_id=f"p{passage_index:03d}", page_number=page_number, text=chunk))
    return passages


def passage_stats(passages: list[Passage]) -> dict[str, Any]:
    lengths = [_estimate_tokens(passage.text) for passage in passages]
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "avg": 0.0, "tiny_passages": 0}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "avg": float(sum(lengths) / len(lengths)),
        "tiny_passages": sum(1 for length in lengths if length <= 12),
    }


def build_document_cache_key(uploaded_file) -> str:
    if uploaded_file is None:
        return "no-pdf"

    payload = {
        "filename": uploaded_file.name,
        "size": uploaded_file.size,
        "sha256": hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
        "embedding_target_tokens": EMBEDDING_TARGET_TOKENS,
        "overlap_sentences": OVERLAP_SENTENCES,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "nli_token_limit": NLI_TOKEN_LIMIT,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cosine_similarity_matrix(embeddings: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0 or np.any(norms == 0):
        return np.zeros(len(embeddings), dtype=float)
    return (embeddings @ query_vector) / (norms * query_norm)


@lru_cache(maxsize=1)
def get_sentence_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_passages(passages: list[Passage], model: SentenceTransformer | None = None) -> np.ndarray:
    if not passages:
        return np.empty((0, 0), dtype=float)
    model = model or get_sentence_model()
    passage_texts = [passage.text for passage in passages]
    embeddings = model.encode(passage_texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=float)


def retrieve_relevant_passages(claims: list[str], passages: list[Passage], embeddings: np.ndarray | None = None) -> dict[str, list[dict[str, Any]]]:
    if not claims:
        return {}
    if not passages:
        return {claim: [] for claim in claims}

    model = get_sentence_model()
    if embeddings is None:
        embeddings = embed_passages(passages, model=model)

    if embeddings.size == 0:
        return {claim: [] for claim in claims}

    output: dict[str, list[dict[str, Any]]] = {}

    for claim in claims:
        query_vector = model.encode(claim, convert_to_numpy=True, show_progress_bar=False)
        similarities = _cosine_similarity_matrix(embeddings, query_vector)
        ranked_indices = np.argsort(similarities)[::-1][:RETRIEVAL_TOP_K]
        matches: list[dict[str, Any]] = []
        for idx in ranked_indices:
            passage = passages[int(idx)]
            matches.append(
                {
                    "passage": passage,
                    "relevance": float(similarities[int(idx)]),
                    "rank": len(matches) + 1,
                }
            )
        output[claim] = matches

    return output


def _aggregate_claim_state(results: list[dict[str, Any]]) -> str:
    supporting = sum(1 for result in results if result.get("nli_result") == "Supported" or result.get("result") == "Supported")
    contradicting = sum(1 for result in results if result.get("nli_result") == "Contradicted" or result.get("result") == "Contradicted")

    if supporting and contradicting:
        return "Mixed evidence found"
    if supporting:
        return "Supporting passage found"
    if contradicting:
        return "Potential contradiction found"
    return "No decisive evidence found"


def assess_claims(claims: list[str], passages_by_claim: dict[str, list[dict[str, Any]]], tokenizer: Any, model: Any) -> list[dict[str, Any]]:
    claim_reviews: list[dict[str, Any]] = []
    for claim in claims:
        retrieved = passages_by_claim.get(claim, [])
        evidence_rows: list[dict[str, Any]] = []
        for retrieved_item in retrieved:
            passage = retrieved_item["passage"]
            nli_result = run_nli_check(passage.text, claim, tokenizer=tokenizer, model=model)
            evidence_rows.append(
                {
                    "passage_id": passage.passage_id,
                    "page_number": passage.page_number,
                    "excerpt": passage.text,
                    "relevance": retrieved_item["relevance"],
                    "nli_result": nli_result["result"],
                    "nli_scores": nli_result["scores"],
                    "raw_label": nli_result["predicted_label"],
                }
            )

        claim_reviews.append(
            {
                "claim": claim,
                "summary": _aggregate_claim_state(evidence_rows),
                "evidence": evidence_rows,
            }
        )
    return claim_reviews


def _check_claim_passage_fit(claim: str, passage_text: str, tokenizer: Any, limit: int = NLI_TOKEN_LIMIT) -> None:
    pair = f"{passage_text} {claim}".strip()
    token_count = len(tokenizer.encode(pair, add_special_tokens=True))
    if token_count > limit:
        raise ValueError(
            "This passage and claim exceed the model's token budget. "
            "Please shorten the claim or upload a smaller document."
        )


def run_paper_review(uploaded_file, claims: list[str], passages: list[Passage] | None = None, embeddings: np.ndarray | None = None) -> dict[str, Any]:
    cleaned_claims = [claim.strip() for claim in claims if claim and claim.strip()]
    if not cleaned_claims:
        raise ValueError("Please enter at least one claim before running the review.")
    if len(cleaned_claims) > MAX_CLAIMS:
        raise ValueError(f"Please enter at most {MAX_CLAIMS} claims per review.")

    if passages is None:
        pages_with_text, total_pages, extracted_pages, empty_pages, partial_warning = extract_pdf_text(uploaded_file)
        passages = build_passages(pages_with_text)
        if not passages:
            raise ValueError("No passages were created from the uploaded PDF.")
        if len(passages) > MAX_PASSAGES:
            raise ValueError(f"This PDF produced too many passages for the demo budget ({MAX_PASSAGES}). Please upload a smaller or more concise document.")
    else:
        total_pages = None
        extracted_pages = None
        empty_pages = 0
        partial_warning = None

    if len(passages) > MAX_PASSAGES:
        raise ValueError(f"This PDF produced too many passages for the demo budget ({MAX_PASSAGES}). Please upload a smaller or more concise document.")

    passages_by_claim = retrieve_relevant_passages(cleaned_claims, passages, embeddings=embeddings)
    tokenizer, model = load_nli_model()
    for claim in cleaned_claims:
        for item in passages_by_claim.get(claim, []):
            _check_claim_passage_fit(claim, item["passage"].text, tokenizer)
    claim_reviews = assess_claims(cleaned_claims, passages_by_claim, tokenizer, model)

    return {
        "filename": uploaded_file.name,
        "page_count": total_pages if total_pages is not None else "n/a",
        "pages_with_text": extracted_pages if extracted_pages is not None else len({p.page_number for p in passages}),
        "empty_pages": empty_pages,
        "passage_count": len(passages),
        "partial_warning": partial_warning,
        "claims": cleaned_claims,
        "reviews": claim_reviews,
    }
