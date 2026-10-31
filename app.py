from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from checker import load_nli_model, run_nli_check
from pdf_review import (
    MAX_CLAIMS,
    build_document_cache_key,
    build_passages,
    embed_passages,
    extract_pdf_text,
    retrieve_relevant_passages,
    run_paper_review,
)

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
EXAMPLES_PATH = Path(__file__).with_name("examples.json")


@st.cache_resource
def get_model():
    tokenizer, model = load_nli_model()
    return tokenizer, model


def load_examples() -> list[dict]:
    with EXAMPLES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_result_from_input(source: str, claim: str) -> dict:
    tokenizer, model = get_model()
    try:
        with st.spinner("Checking the claim against the excerpt..."):
            return run_nli_check(source, claim, tokenizer=tokenizer, model=model)
    except Exception as exc:  # pragma: no cover - surfaced in Streamlit UI
        st.error(str(exc))
        return {}


def parse_claims(raw_claims: str) -> list[str]:
    return [line.strip() for line in raw_claims.splitlines() if line and line.strip()]


st.set_page_config(page_title="Research Claim Checker", page_icon="🧪")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1060px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.22rem 0.6rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        border: 1px solid transparent;
        margin-right: 0.5rem;
    }
    .support-badge {
        background: #eaf6ee;
        border-color: #bfe0c9;
        color: #214a36;
    }
    .contradict-badge {
        background: #fbe9e7;
        border-color: #e7b5b0;
        color: #7d352d;
    }
    .neutral-badge {
        background: #f6f0e7;
        border-color: #e0d2b8;
        color: #5b4a30;
    }
    .quote-box {
        background: #fffdfb;
        border: 1px solid #d7cec5;
        border-radius: 0.75rem;
        padding: 0.9rem 1rem;
    }
    .small-copy {
        color: #5f5a56;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Research Claim Checker")
st.caption("Find the evidence behind a claim.")

pdf_tab, manual_tab = st.tabs(["Paper review", "Check an excerpt"])

with manual_tab:
    examples = load_examples()
    example_labels = [item["label"] for item in examples]
    selected_example = st.selectbox(
        "Load an example",
        ["Custom input"] + example_labels,
        index=0,
    )

    source_key = "source_excerpt"
    claim_key = "claim_text"

    if selected_example != "Custom input":
        example = next(item for item in examples if item["label"] == selected_example)
        if st.session_state.get("loaded_example_label") != selected_example:
            st.session_state[source_key] = example["source"]
            st.session_state[claim_key] = example["claim"]
            st.session_state["loaded_example_label"] = selected_example
            st.session_state.pop("result", None)
            st.session_state.pop("last_signature", None)
    else:
        st.session_state["loaded_example_label"] = "Custom input"

    source = st.text_area("Source passage", key=source_key, height=220)
    claim = st.text_area("Claim", key=claim_key, height=120)

    form = st.form("claim_check_form")
    with form:
        submitted = st.form_submit_button("Check claim")

    if submitted:
        current_signature = (source.strip(), claim.strip())
        previous_signature = st.session_state.get("last_signature")
        if previous_signature != current_signature:
            st.session_state.pop("result", None)
        st.session_state["last_signature"] = current_signature

        result = set_result_from_input(source, claim)
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if result:
        status = result["result"]
        if status == "Supported":
            st.markdown(
                "<div style='padding:12px 14px;border-radius:12px;background:#e8f5e9;border:1px solid #2e7d32;color:#1b5e20;font-weight:600;'>✅ Supported</div>",
                unsafe_allow_html=True,
            )
            explanation = "The claim is entailed by the provided excerpt: the excerpt supports the claim as written."
        elif status == "Contradicted":
            st.markdown(
                "<div style='padding:12px 14px;border-radius:12px;background:#ffebee;border:1px solid #c62828;color:#b71c1c;font-weight:600;'>❌ Contradicted</div>",
                unsafe_allow_html=True,
            )
            explanation = "The excerpt conflicts with the claim: the claim does not match the information provided."
        else:
            st.markdown(
                "<div style='padding:12px 14px;border-radius:12px;background:#fff8e1;border:1px solid #f9a825;color:#6d4c41;font-weight:600;'>⚠️ Insufficient evidence</div>",
                unsafe_allow_html=True,
            )
            explanation = "The excerpt is not strong enough to support the claim, either because it is neutral or because it does not address the claim directly."

        st.write(explanation)

        with st.expander("Technical details"):
            st.write("Model scores are not calibrated confidence.")
            for label, score in result["scores"].items():
                st.write(f"{label}: {score:.4f}")
    else:
        st.info("Enter a source passage and a claim, then click “Check claim.”")

    st.markdown("---")

with pdf_tab:
    uploaded_file = st.file_uploader("Paper", type=["pdf"], help="Text-based PDF · up to 30 pages · 10 MB", key="paper_pdf")

    claims_input = st.text_area(
        "Claims",
        placeholder="Claim 1\nClaim 2\nClaim 3",
        height=170,
        key="paper_claims",
        help="One claim per line · up to 5 claims",
    )

    claims = parse_claims(claims_input)
    current_document_key = build_document_cache_key(uploaded_file) if uploaded_file else None
    current_signature = (current_document_key, tuple(claims))
    previous_signature = st.session_state.get("paper_review_signature")
    if previous_signature != current_signature:
        st.session_state.pop("paper_review_result", None)
        st.session_state.pop("paper_review_cache_key", None)
        st.session_state["paper_review_signature"] = current_signature

    analyze_clicked = st.button("Review claims", use_container_width=True)

    if analyze_clicked:
        st.session_state.pop("paper_review_error", None)
        if uploaded_file is None:
            st.session_state["paper_review_error"] = "Please upload a PDF before analyzing claims."
            st.error(st.session_state["paper_review_error"])
        elif not claims:
            st.session_state["paper_review_error"] = "Please enter at least one claim before running the review."
            st.error(st.session_state["paper_review_error"])
        elif len(claims) > MAX_CLAIMS:
            st.session_state["paper_review_error"] = f"Please enter at most {MAX_CLAIMS} claims per run."
            st.error(st.session_state["paper_review_error"])
        else:
            try:
                doc_key = build_document_cache_key(uploaded_file)
                previous_key = st.session_state.get("paper_document_key")
                result_cache_key = (doc_key, tuple(claims))
                document_cache = st.session_state.setdefault("paper_document_cache", {})
                cached_result = st.session_state.get("paper_review_result")
                same_result_cached = cached_result is not None and st.session_state.get("paper_review_cache_key") == result_cache_key

                if previous_key == doc_key and doc_key in document_cache:
                    doc_entry = document_cache[doc_key]
                    passages = doc_entry["passages"]
                    embeddings = doc_entry["embeddings"]
                    status_message = "Reusing document embeddings for the same PDF..."
                    extraction_ms = 0.0
                    passage_ms = 0.0
                    embedding_ms = 0.0
                else:
                    start_extraction = time.perf_counter()
                    pages_with_text, total_pages, extracted_pages, empty_pages, partial_warning = extract_pdf_text(uploaded_file)
                    extraction_ms = (time.perf_counter() - start_extraction) * 1000.0
                    start_passages = time.perf_counter()
                    passages = build_passages(pages_with_text)
                    passage_ms = (time.perf_counter() - start_passages) * 1000.0
                    start_embedding = time.perf_counter()
                    embeddings = embed_passages(passages)
                    embedding_ms = (time.perf_counter() - start_embedding) * 1000.0
                    st.session_state["paper_document_key"] = doc_key
                    document_cache[doc_key] = {
                        "passages": passages,
                        "embeddings": embeddings,
                        "summary": {
                            "page_count": total_pages,
                            "pages_with_text": extracted_pages,
                            "empty_pages": empty_pages,
                            "partial_warning": partial_warning,
                        },
                    }

                if same_result_cached:
                    st.session_state["paper_review_result"]["timings"] = {
                        **st.session_state["paper_review_result"].get("timings", {}),
                        "result_cache_hit": True,
                    }
                else:
                    start_retrieval = time.perf_counter()
                    passages_by_claim = retrieve_relevant_passages(claims, passages, embeddings=embeddings)
                    retrieval_ms = (time.perf_counter() - start_retrieval) * 1000.0
                    start_nli = time.perf_counter()
                    review_result = run_paper_review(uploaded_file, claims, passages=passages, embeddings=embeddings)
                    nli_ms = (time.perf_counter() - start_nli) * 1000.0
                    review_result["timings"] = {
                        "doc_key": doc_key,
                        "model_load_ms": 0.0,
                        "extraction_ms": extraction_ms,
                        "passage_ms": passage_ms,
                        "embedding_ms": embedding_ms,
                        "retrieval_ms": retrieval_ms,
                        "nli_ms": nli_ms,
                        "result_cache_hit": False,
                    }
                    st.session_state["paper_review_result"] = review_result
                    st.session_state["paper_review_cache_key"] = result_cache_key
                    st.session_state.pop("paper_review_error", None)
                st.success("Review complete.")
            except Exception as exc:
                st.session_state["paper_review_error"] = str(exc)
                st.error(str(exc))

    result = st.session_state.get("paper_review_result")
    if result:
        st.subheader("Results")
        summary_text = f"{result['filename']} · {result['page_count']} pages · {result['passage_count']} passages"
        st.caption(summary_text)
        if result.get("partial_warning"):
            st.warning(result["partial_warning"])

        for index, review in enumerate(result["reviews"], start=1):
            if review["summary"] == "Supporting passage found":
                label = "Support found"
                badge_class = "support-badge"
            elif review["summary"] == "Potential contradiction found":
                label = "Possible contradiction"
                badge_class = "contradict-badge"
            elif review["summary"] == "Mixed evidence found":
                label = "Mixed evidence"
                badge_class = "neutral-badge"
            else:
                label = "No decisive evidence"
                badge_class = "neutral-badge"

            with st.expander(f"{index}. {review['claim']}", expanded=True):
                st.markdown(f"<span class='status-badge {badge_class}'>{label}</span>", unsafe_allow_html=True)
                if not review["evidence"]:
                    st.write("No relevant passages were retrieved for this claim.")
                    continue

                ordered_evidence = sorted(
                    review["evidence"],
                    key=lambda item: (item["nli_result"] not in {"Supported", "Contradicted"}, item["relevance"]),
                    reverse=True,
                )

                for item in ordered_evidence:
                    status_label = item["nli_result"]
                    if status_label == "Supported":
                        badge = "Support found"
                        badge_class = "support-badge"
                    elif status_label == "Contradicted":
                        badge = "Contradiction"
                        badge_class = "contradict-badge"
                    else:
                        badge = "Neutral"
                        badge_class = "neutral-badge"
                    st.markdown(f"<span class='status-badge {badge_class}'>{badge}</span> <span class='small-copy'>PDF page {item['page_number']}</span>", unsafe_allow_html=True)
                    st.caption(f"Passage ID: {item['passage_id']} · Relevance: {item['relevance']:.4f}")
                    st.markdown(f"<div class='quote-box'>{item['excerpt'].replace(chr(10), '<br><br>')}</div>", unsafe_allow_html=True)
                    with st.expander("Technical details"):
                        st.write("Model scores are not calibrated confidence.")
                        for label, score in item["nli_scores"].items():
                            st.write(f"{label}: {score:.4f}")


    elif st.session_state.get("paper_review_error"):
        st.warning(f"The previous run failed: {st.session_state['paper_review_error']}")
    elif uploaded_file is not None and claims:
        st.info("The review is ready to run. Click “Analyze claims” to compare each claim against the extracted evidence.")

    st.markdown("---")
    with st.expander("About this tool"):
        st.write("- Finds and checks selected passages, not every possible interpretation of the paper.")
        st.write("- Missing evidence may reflect retrieval or model errors.")
        st.write("- Does not verify source credibility or establish factual truth.")
        st.write("- Supports text-based PDFs; scanned documents require OCR, which is not included.")
    st.caption("Checks the supplied text. Results can be wrong.")
