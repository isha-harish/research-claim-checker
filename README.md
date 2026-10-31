# Research Claim Checker

Find relevant passages in a research paper and check whether they support a claim.

**[Open the live app](https://research-claim-checker-d4buzzc9zenu6etkd8aozh.streamlit.app/)**

Built for the MiraFlores AI engineering case study.

## What it does

### Paper review

1. Upload a text-based PDF.
2. Enter up to five claims, one per line.
3. Review the retrieved evidence and model assessments, with PDF page references.

The app returns a review result for each claim:

- **Support found**
- **Possible contradiction**
- **Mixed evidence**
- **No decisive evidence**

Each retrieved passage also has its own assessment. All retrieved evidence remains available for inspection.

### Check an excerpt

Paste a source passage and one claim to compare them directly, without PDF extraction or retrieval.

Results are **Supported**, **Contradicted**, or **Insufficient evidence**.

The app checks relationships between claims and supplied text. It does not verify source credibility or determine objective truth.

## Quick start

The local environment was tested with **Python 3.13.5**. Dependencies are pinned in [requirements.txt](requirements.txt).

```bash
git clone https://github.com/isha-harish/research-claim-checker.git
cd research-claim-checker

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m streamlit run app.py
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Open the local URL printed in the terminal. If port 8501 is occupied:

```bash
python -m streamlit run app.py --server.port 8502
```

The first model load requires internet access to download files from Hugging Face. Downloads are cached locally; startup can take longer than subsequent reviews. No paid inference API or API key is required.

## Example inputs and expected results

### Paper review

Download [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) and upload the PDF.

Enter these claims, one per line:

```text
The Transformer architecture removes recurrence and convolutions and relies on attention mechanisms.
The Transformer uses recurrent layers to process sequences.
The paper evaluates the Transformer in a clinical trial involving patients.
```

Expected review results:

| Claim | Expected result |
|---|---|
| Removes recurrence and convolutions | Support found |
| Uses recurrent layers | Possible contradiction |
| Evaluated in a clinical trial | No decisive evidence |

These overall results were observed in browser testing. Individual retrieved passages can still receive incorrect labels.

Page references use the PDF's page order, which may differ from printed page numbers.

### Check an excerpt

Synthetic source passage:

```text
The study tested the device on 120 adults. Participants used it for eight weeks. Sleep duration increased, but the study did not measure anxiety.
```

Try each claim separately:

| Claim | Expected result |
|---|---|
| The study tested the device on adults. | Supported |
| The study tested the device only on children. | Contradicted |
| The device reduced anxiety. | Insufficient evidence |

For an invalid-input check, leave the source passage empty and submit. The app should display a validation error.

Expected labels are test expectations, not guaranteed model outputs.

## How it works

```text
PDF extraction
    → page-aware passage construction
    → semantic retrieval
    → passage-level NLI comparison
    → evidence review
```

1. **Extraction:** pypdf extracts text while retaining PDF page numbers.
2. **Passages:** text is packed into token-bounded passages within page boundaries.
3. **Retrieval:** sentence embeddings identify the three most relevant passages for each claim.
4. **Comparison:** an NLI model compares each retrieved passage with the claim.
5. **Review:** the app combines passage assessments into a qualified review result and displays the evidence.

Retrieval similarity measures relevance, not support. NLI scores are model outputs, not calibrated confidence or probabilities that a claim is true.

### Review aggregation

| Assessments among retrieved passages | Review result |
|---|---|
| Supporting and contradicting passages | Mixed evidence |
| Supporting passages, no contradicting passages | Support found |
| Contradicting passages, no supporting passages | Possible contradiction |
| Neither supporting nor contradicting passages | No decisive evidence |

“No decisive evidence” can mean the excerpt does not establish the claim, retrieval missed relevant evidence, or the model misclassified it. It does not mean the claim is false.

## Models

| Task | Model | License |
|---|---|---|
| Passage retrieval | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache 2.0 |
| Natural language inference | [cross-encoder/nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small) | Apache 2.0 |

The source passage is passed as the NLI premise and the claim as the hypothesis.

NLI labels map to:

- `entailment` → Supported
- `contradiction` → Contradicted
- `neutral` → Insufficient evidence

Both models run on CPU. No training or fine-tuning is performed.

## Supported inputs and limits

- One PDF per review
- Text-based PDFs only; no OCR
- Maximum file size: 10 MB
- Maximum document length: 30 PDF pages
- One to five claims per run
- Short English claims and passages within model token limits
- An additional passage-count budget enforced by the application

Invalid, empty, encrypted, and PDFs without usable extracted text are rejected. Inputs exceeding processing limits receive an error rather than being silently truncated.

PDF extraction may produce imperfect reading order, especially around columns, equations, tables, and footnotes.

## Caching and privacy

Models and tokenizers are reused within the application process.

Extracted pages, passages, and document embeddings are retained in session state, keyed by PDF content and processing settings.

- Changing claims clears previous results but retains the document index.
- Changing the PDF invalidates the document index and results.
- New claims still require retrieval and inference.
- Uploaded document content is not placed in a global cross-user cache.
- The application does not intentionally persist uploaded PDFs or claims to a database.

On the hosted app, uploaded files are processed on the hosting server. Inference does not send their contents to a paid external model API.

## Evaluation

Run the preserved diagnostic scripts:

```bash
python evaluate.py
python research_evaluate.py
python harder_research_evaluate.py
```

Recorded results for the excerpt-comparison component:

| Diagnostic set | Result |
|---|---|
| Original synthetic examples | 8/12 |
| Initial paper-excerpt examples | 7/7 |
| Final harder paper-excerpt examples | 3/5 |

These small, hand-built sets include limited coverage and some debatable expected labels. They are not independent benchmarks or accuracy estimates for the complete PDF pipeline.

The original synthetic set includes an ambiguous example: discontinuing a product does not necessarily establish that remaining inventory could not be sold later.

The first harder-set run scored 4/5. One case was subsequently replaced to better satisfy the source requirements; the revised set scored 3/5. The earlier record and result are preserved in [harder_research_historical.json](harder_research_historical.json).

### Full PDF workflow checks

The complete 15-page *Attention Is All You Need* PDF was processed into 83 passages after correcting an earlier fragmentation issue.

Browser checks covered:

- A supported claim
- A claim conflicting with the paper
- An unrelated clinical-trial claim
- Evidence display and page references
- Manual excerpt checking
- Blank-input validation
- Clearing outdated results after input changes
- Reusing the document index for subsequent claims

The public deployment was also tested manually.

### Observed model failures

For the claim that the Transformer uses recurrent layers:

- A page-5 passage describing attention was labeled contradictory without explicitly excluding recurrence.
- A page-2 passage explicitly excluding sequence-aligned RNNs was labeled neutral.

These errors remain documented. No example-specific rules were added to hide them.

## Project files

| File | Purpose |
|---|---|
| [app.py](app.py) | Streamlit interface and session state |
| [checker.py](checker.py) | NLI loading, validation, and inference |
| [pdf_review.py](pdf_review.py) | Extraction, passage construction, retrieval, and aggregation |
| [requirements.txt](requirements.txt) | Pinned dependencies |
| [.streamlit/config.toml](.streamlit/config.toml) | Application theme |
| [examples.json](examples.json) | Synthetic excerpt examples |
| [evaluate.py](evaluate.py) | Synthetic diagnostic evaluator |
| [research_examples.json](research_examples.json) | Initial paper-excerpt examples |
| [research_evaluate.py](research_evaluate.py) | Initial paper-excerpt evaluator |
| [harder_research_examples.json](harder_research_examples.json) | Harder paper-excerpt examples |
| [harder_research_evaluate.py](harder_research_evaluate.py) | Harder-set evaluator |
| [harder_research_historical.json](harder_research_historical.json) | Historical evaluation record |

## Deployment

The app is deployed on **Streamlit Community Cloud**:

**https://research-claim-checker-d4buzzc9zenu6etkd8aozh.streamlit.app/**

To deploy your own copy:

1. Fork or copy this repository into your GitHub account.
2. Sign in to Streamlit Community Cloud.
3. Create an app using your repository and branch.
4. Set the entry point to `app.py`.
5. Select a Python version compatible with the pinned dependencies.
6. Deploy without adding API secrets.

The repository includes the theme configuration. Credentials, virtual environments, caches, and model weights should remain excluded from Git.

## Limitations and deferred work

- Scanned PDFs and image-only evidence are not supported.
- Charts, figures, and table structure are not interpreted by a vision model.
- Retrieval can miss relevant evidence.
- NLI can misclassify negation, numerical comparisons, qualifications, and indirect statements.
- Evidence is checked passage by passage; reasoning across passages is not implemented.
- Claims must be entered individually.
- Multiple PDFs, web research, automatic claim extraction, accounts, and persistent history are not implemented.

With additional time and budget, possible extensions include web research, semantic search across multiple papers, and vision-based review of figures and tables. These would require separate evaluation.

## AI assistance and reuse

The application uses pretrained Hugging Face models and open-source libraries. Model predictions are not mocked. Synthetic test examples are labeled, and no source code was reused.
