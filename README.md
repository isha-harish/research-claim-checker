# Research Claim Checker

## Purpose
This project is a local Streamlit app for evidence-assisted claim checking against a PDF or a short excerpt. It is designed for a researcher or reviewer who wants to compare a claim to retrieved evidence and see whether the text supports, contradicts, or fails to establish the claim.

This app is intentionally not a paper-level truth detector and does not guarantee factual correctness. It is a qualitative evidence review tool using small local models.

## Current app behavior
The app has two tabs:

1. Manual excerpt checker
   - paste a short source passage and a claim
   - run a real NLI model against that excerpt
   - see a Supported / Contradicted / Insufficient evidence result

2. Paper review
   - upload one text-based PDF
   - enter one to five claims, one per line
   - extract page text and build passage chunks
   - retrieve the most relevant passages for each claim
   - run NLI on those retrieved passages
   - review evidence with page numbers and excerpts

The app deliberately separates:
- retrieval relevance: the passage may be relevant
- NLI comparison: the passage supports or contradicts the claim
- overall paper truth: not asserted by the app

## Architecture
Pipeline:

PDF extraction → passage construction → retrieval → NLI comparison → evidence review

Key files:
- [app.py](app.py): Streamlit UI and session-state handling
- [checker.py](checker.py): NLI model loading, validation, and label mapping
- [pdf_review.py](pdf_review.py): PDF extraction, passage packing, retrieval, claim aggregation, and per-document caching
- [examples.json](examples.json): example excerpt-checking demos
- [evaluate.py](evaluate.py): preserved diagnostic evaluation for the manual excerpt workflow

## Models in use
### NLI model
- Model: cross-encoder/nli-deberta-v3-small
- Source: https://huggingface.co/cross-encoder/nli-deberta-v3-small
- License: Apache 2.0
- Label mapping used in the app:
  - entailment → Supported
  - contradiction → Contradicted
  - neutral → Insufficient evidence

### Retrieval model
- Model: sentence-transformers/all-MiniLM-L6-v2
- Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- License: Apache 2.0
- Used for passage similarity search before NLI comparison.

## Setup
Tested with Python 3.13.5 and the project venv.

```bash
cd /Users/ishaharish/Desktop/project_isha_research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the app:

```bash
cd /Users/ishaharish/Desktop/project_isha_research
source .venv/bin/activate
streamlit run app.py --server.port 8501
```

If port 8501 is already in use, start on another free port:

```bash
streamlit run app.py --server.port 8503
```

## Dependencies and version pins
The project keeps dependency versions pinned in [requirements.txt](requirements.txt). This is intentional for reproducibility and local testing.

## Supported inputs and limits
- One PDF at a time
- Text-based PDFs only
- Maximum file size: 10 MB
- Maximum PDF pages: 30
- Maximum claims per run: 5
- Model downloads are cached locally by Hugging Face on first use

The app rejects unreadable, encrypted, empty, or image-only/no-text PDFs with clear errors. It does not run OCR.

## Caching behavior
The app intentionally caches:
- the NLI tokenizer/model once per process
- the sentence encoder model once per process
- the extracted PDF pages, passages, and document embeddings within a session, keyed by a hash of PDF bytes and processing settings

This means:
- changing claims clears results but retains the document index
- changing the PDF invalidates the document index and results
- repeated claims still require retrieval/NLI unless the exact result is cached
- uploaded PDF content is not globally shared across users

## Evaluation data and honesty
The repository preserves the original evaluation scripts and example datasets. They are kept as diagnostic artifacts and are not treated as a claim that the system is production-grade.

The app was validated locally with concrete checks for:
- PDF extraction and page retention
- text-to-claim comparison on example inputs
- empty/no-evidence handling
- invalid input handling
- stale-result invalidation when claims or PDFs change

## Known limitations
- OCR is intentionally not supported.
- Scanned PDFs with no extractable text will fail clearly.
- Retrieval can miss relevant passages when phrasing differs from the paper text.
- NLI can still be wrong on nuanced reasoning, arithmetic, definitions, or indirect statements.
- The app is demo-quality evidence review, not a validated research-grade claim discriminator.

## Deployment status
Local app execution in this workspace is working through the Streamlit UI.

The current deployment blocker is not a code bug in the app itself; the repo has not been pushed to a live GitHub or hosting target in this environment. A hosted deployment would require a repository push and platform configuration outside this local workspace.

## Implementation status
Implemented:
- text-based PDF upload
- page-aware extraction
- passage packing and token-aware chunking
- retrieval by sentence embeddings
- NLI-based claim comparison
- evidence review with page references
- validation and clear errors
- manual excerpt checker preserved as a secondary tab

Not included:
- OCR
- multiple PDFs in one run
- automatic claim extraction from arbitrary prose
- repository-level deployment configuration
- user accounts or persistent storage
- model fine-tuning or external API integrations

## AI-assistance disclosure
This project uses real local Hugging Face models and CPU inference. No paid APIs, persistent data store, or external service key was required for the local workflow.
