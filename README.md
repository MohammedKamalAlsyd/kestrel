# Kestrel

**Kestrel** is a 350M–700M parameter modern decoder-only LLM engineered to outperform OpenAI's 1.5B **GPT-2 XL** across classic GPT-2 language modeling (**WikiText-103**, **LAMBADA**) and modern commonsense & reasoning benchmarks (**HellaSwag**, **PIQA**, **WinoGrande**, **ARC-Easy**, **ARC-Challenge**).

---

## Pretraining Dataset Decision: `Nemotron-ClimbMix (400B)`

Kestrel trains on **one single corpus from start to finish**—**`Nemotron-ClimbMix (400B)`** ([`nvidia/Nemotron-ClimbMix`](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) / [`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle))—selected without test-set peeking by pairing our internal corpus audit with NVIDIA's **CLIMB** scaling results ([*Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training*, arXiv:2504.13161](https://arxiv.org/pdf/2504.13161)).

### Why `Nemotron-ClimbMix (400B)`?

1. **Strongest Controlled Scaling Performance (CLIMB, arXiv:2504.13161):**
   - **Beats every open competitor under equal token budgets:** When pretraining a 1B model from scratch across 32B–400B tokens, `ClimbMix` outperforms **`FineWeb-Edu`**, **`DCLM-Baseline`**, **`SmolLM`**, **`Nemotron-CC-HQ`**, and random clustering (`ClimbLab-Random`), reaching **`52.43%`** average performance across 12 downstream benchmarks at 400B tokens (compared to `~50.2%` for `DCLM-Baseline` and `~49.4%` for `FineWeb-Edu`).
   - **Built specifically for compute-efficient sub-1B/1B models:** CLIMB embeds web data, clusters it into 20 semantic groups, and uses lightweight proxy models + regression-based iterative bootstrapping (`LightGBM`) to discover an optimal 400B-token training mixture—boosting a continuously trained 1B model **+2.0% over `Llama-3.2-1B`** across downstream evaluations.
2. **Backed by Our Own Corpus Audit (`notebooks/01_dataset_audit.ipynb`):**
   - **High reasoning & science density without test-set peeking:** In our shared `all-MiniLM-L6-v2` BERT topic map (`K = 12`), `Nemotron-ClimbMix` concentrates **`60.6%`** of its documents into core science/health/climate (**`38.2%`**) and practical Q&A/how-to explanations (**`22.4%`**), compared to **`7.6%`** core science in `OpenWebText` (GPT-2's distribution, which spends `42.0%` on 2016–2019 news and politics).
   - **Clean single-dataset simplicity:** Compact 400B-token footprint with short, dense documents (`632.0` mean GPT-2 tokens/doc, `541.0` median), eliminating multi-stage curriculum complexity.
3. **Known Trade-Offs Noted Upfront:**
   - **License:** Upstream NVIDIA `Nemotron-ClimbMix` carries **`CC-BY-NC-4.0`** (non-commercial research use); our primary commercially permissive backup from the audit is **`DCLM-Baseline 1.0`** (`CC-BY-4.0`).
   - **Formatting:** Occasional raw HTML list tags (`<ul><li>`) appeared in random spot-checks—worth keeping in mind during data cleaning/tokenization if further filtering is needed.

---

## Dataset Audit & EDA Workflow

- **Kaggle-Ready Self-Contained Audit Notebook:** [`notebooks/01_dataset_audit.ipynb`](notebooks/01_dataset_audit.ipynb) — modular, strictly test-set-blind corpus audit featuring multi-shard randomized streaming (`3.5M` GPT-2 tokens across 8 random shards per corpus), exploratory data analysis (lengths, character formatting, TF-IDF terms, raw spot-checks), and BERT semantic topic modeling (`all-MiniLM-L6-v2` + `google/flan-t5-small` full-width topic heatmap).
- **Shared Utilities:** [`src/kestrel/utils/`](src/kestrel/utils/) (`notebook.py`, `text.py`) — reusable Kaggle/Hugging Face cache helpers and GPT-2 tokenization/text helpers shared across Kestrel notebooks.
- **Empirical Audit Notes:** [`docs/dataset_justification.md`](docs/dataset_justification.md) — full audited corpus summary table, BERT topic breakdown, and published scaling comparisons (CLIMB, DCLM, FineWeb-Edu).
- **Candidate Architectures & Evaluation Protocol:** [`docs/architectures.md`](docs/architectures.md) — full macro + repeated-block + attention/mixer/MoE diagrams for all 4 candidates (`configs/model/*.json`) and our 2-stage evaluation methodology vs. GPT-2 XL.

## Quickstart

```bash
# Install with data & topic-modeling dependencies
uv sync --extra data
```

On **Kaggle**, import [`notebooks/01_dataset_audit.ipynb`](notebooks/01_dataset_audit.ipynb), enable **Internet** (and optionally a **T4 GPU** for faster BERT encoding) in the right-hand sidebar, and click **Run All**.
