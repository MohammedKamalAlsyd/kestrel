# Kestrel

**Kestrel** is a 350M–700M parameter modern decoder-only LLM engineered to outperform OpenAI's 1.5B **GPT-2 XL** across classic GPT-2 language modeling (**WikiText-103**, **LAMBADA**) and modern commonsense & reasoning benchmarks (**HellaSwag**, **PIQA**, **WinoGrande**, **ARC-Easy**, **ARC-Challenge**).

## Dataset Audit & EDA Workflow

- **Kaggle-Ready Self-Contained Audit Notebook:** [`notebooks/01_dataset_audit.ipynb`](notebooks/01_dataset_audit.ipynb) — strictly test-set-blind corpus audit featuring multi-shard randomized streaming (`3.5M` GPT-2 tokens across 8 random shards per corpus), exploratory data analysis (lengths, character formatting, TF-IDF terms, raw spot-checks), and BERT semantic topic modeling (`all-MiniLM-L6-v2` + `google/flan-t5-small` full-width topic heatmap).
- **Shared Utilities:** [`src/kestrel/utils/`](src/kestrel/utils/) (`notebook.py`, `text.py`) — reusable Kaggle/Hugging Face cache helpers and GPT-2 tokenization/text helpers shared across Kestrel notebooks.
- **Empirical Audit Notes:** [`docs/dataset_justification.md`](docs/dataset_justification.md) — corpus audit summary paired with published pretraining scaling curves (CLIMB, DCLM, FineWeb-Edu) and single-dataset trade-off notes.

## Quickstart

```bash
# Install with data & topic-modeling dependencies
uv sync --extra data
```

On **Kaggle**, import [`notebooks/01_dataset_audit.ipynb`](notebooks/01_dataset_audit.ipynb), enable **Internet** (and optionally a **T4 GPU** for faster BERT encoding) in the right-hand sidebar, and click **Run All**.
