# Kestrel

**Kestrel** is a 350M–700M parameter modern decoder-only LLM engineered to outperform OpenAI's 1.5B **GPT-2 XL** across all core GPT-2 benchmarks (**WikiText-103**, **LAMBADA**, **HellaSwag**, and **ARC-Easy**) using modern data curation, two-stage WSD annealing, and strict zero-leakage decontamination.

## Dataset Audit & Selection Workflow

![Kestrel Data Selection Pipeline](docs/images/01_kestrel_data_pipeline.png)

- **Reusable Diagnostics Module:** [`src/kestrel/data/diagnostics.py`](src/kestrel/data/diagnostics.py) — equal-token streaming (`tiktoken` `gpt2`), stratified content vs. function word-transition networks & conditional entropy, document-level LZ4 Goldilocks (`0.65–0.80`) compression & Gopher/FineWeb 2–10-gram repetition, OpenWebText & benchmark JSD/Jaccard alignment, and exact 13-gram contamination checks.
- **Kaggle-Ready Interactive Audit Notebook:** [`notebooks/01_dataset_audit.ipynb`](notebooks/01_dataset_audit.ipynb) — self-contained with beginner-friendly explanations, explanatory diagrams, animated walkthroughs, and automatic GitHub repo sync on Kaggle (`git clone --depth 1 https://github.com/MohammedKamalAlsyd/kestrel.git`).
- **Mixture Justification Report:** [`docs/dataset_justification.md`](docs/dataset_justification.md) — full comparison of all 6 audited corpora (`FineWeb-Edu`, `DCLM-Baseline 1.0`, `Nemotron-CC v2.1`, `Nemotron-ClimbMix`, `FinePDFs-Edu`, `OpenWebText` control) and exact Stage-1 (90%) + Stage-2 Annealing (10%) mixture ratios.

## Quickstart

```bash
# Install with data audit & test dependencies
uv sync --extra data --extra dev

# Run diagnostics unit tests
PYTHONPATH=src pytest -v tests/test_diagnostics.py
```

On **Kaggle**, upload [`notebooks/01_dataset_audit.ipynb`](notebooks/01_dataset_audit.ipynb), enable **Internet** in the Kaggle sidebar, and run all cells — Cell 1 automatically clones this repository and installs missing dependencies.
