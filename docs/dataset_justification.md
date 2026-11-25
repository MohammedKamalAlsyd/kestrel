# Kestrel Pretraining Dataset Audit — Empirical Findings

> **Goal:** Audit modern open pretraining corpora against **OpenWebText** (GPT-2's 2019 distribution)—strictly **without peeking at held-out evaluation benchmarks**—and combine those findings with published scaling curves to pick **one clean, high-signal dataset** for training Kestrel (350M–700M).

All corpus statistics below come directly from our multi-shard randomized run of [`notebooks/01_dataset_audit.ipynb`](../notebooks/01_dataset_audit.ipynb) (`8` random remote shards per dataset, `shuffle_buffer_size=2,000`, up to `3.5M` GPT-2 tokens / `5,000` docs per corpus).

---

## 1. Audited Candidate Pool & Corpus Summary

| Dataset | HF Repo & License | Docs | Tokens | Median Tok/Doc | Mean Tok/Doc | P99 Tok/Doc | Newlines / 1k | Topic Entropy (bits) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FineWeb-Edu (10BT)** | `HuggingFaceFW/fineweb-edu` (`ODC-By v1.0`) | 3,319 | 3,501,102 | 644.0 | 1,054.9 | 7,930.6 | **4.53** | **3.437** |
| **DCLM-Baseline 1.0** | `mlfoundations/dclm-baseline-1.0-parquet` (`CC-BY-4.0`) | 2,734 | 3,500,699 | 674.5 | 1,280.4 | 12,961.3 | 12.26 | 3.373 |
| **Nemotron-ClimbMix (400B)** | `karpathy/climbmix-400b-shuffle` (`CC-BY-NC-4.0` upstream) | 5,000 | 3,160,216 | 541.0 | 632.0 | 3,871.3 | 7.57 | 3.233 |
| **FinePDFs-Edu (`eng_Latn`)** | `HuggingFaceFW/finepdfs-edu` (`ODC-By v1.0`) | 1,080 | 3,500,022 | 1,356.0 | 3,240.8 | 30,095.2 | 17.43 | 3.136 |
| **OpenWebText** *(Control)* | `Skylion007/openwebtext` (`CC0 / PD`) | 3,160 | 3,500,186 | 724.5 | 1,107.7 | 6,965.8 | 8.87 | 3.049 |

---

## 2. Key Insights: EDA, BERT Topics & Published Scaling Curves

Using `all-MiniLM-L6-v2` embeddings + KMeans (`K = 12`) + keyword-only `google/flan-t5-small` labeling across 2,500 sampled documents (`500` per corpus):

1. **Why GPT-2 XL (`OpenWebText`) Leaves Room to Beat on Reasoning:**
   - **42.0%** of `OpenWebText` falls into US politics, elections, and military news (`T00: 28.6%` + `T04: 13.4%`), and only **7.6%** covers core science, medicine, and climate (`T01 + T05 + T10`), yielding the lowest topic entropy (`3.049 bits`).
2. **Published Controlled Pretraining Hierarchy (Zero Test-Set Peeking):**
   - Across published controlled pretraining runs, the empirical scaling ladder is **`Nemotron-ClimbMix` > `DCLM-Baseline 1.0` > `FineWeb-Edu` >> legacy web corpora (`FineWeb`, `C4`, `Dolma`, `The Pile`, `OpenWebText`)**:
     - **CLIMB (1B, 32B–400B tokens):** `ClimbMix` (`52.43%`) > `DCLM-Baseline` (`~50.2%`) > `FineWeb-Edu` (`~49.4%`).
     - **DCLM (7B, 140B & 280B tokens):** `DCLM-Baseline` decisively beats `FineWeb-Edu` — `48.9` vs. `41.9` CORE (+7.0), `50.8` vs. `37.3` MMLU (+13.5), and `31.8` vs. `24.5` EXTENDED (+7.3) at 280B tokens.
     - **FineWeb-Edu (1.8B, 350B tokens):** Outperforms earlier 2023–2024 web corpora (`FineWeb`, `C4`, `Dolma`, `RefinedWeb`, `SlimPajama`, `The Pile`, `RedPajama2`) on 8 NLP benchmarks, MMLU, and ARC.
3. **Corpus Profiles:**
   - **`Nemotron-ClimbMix (400B)`:** Concentrates **60.6%** into science (`38.2%`) and practical Q&A/how-to (`T08: 22.4%`); shorter docs (`632` tok/doc mean) with occasional raw HTML (`<ul><li>`) and upstream **`CC-BY-NC-4.0`** non-commercial terms.
   - **`DCLM-Baseline 1.0`:** Strongest commercially permissive (`CC-BY-4.0`) pretraining track record; rich general narrative (`T09: 21.0%`) & tech (`T06: 12.4%`), fluent long-form articles (`1,280` tok/doc), and broad topic diversity (`3.373` bits).
   - **`FineWeb-Edu (10BT)`:** Highest topic breadth (`3.437 bits`), cleanest paragraph prose (`4.53` newlines/1k chars), `37.4%` science/health/climate, permissive **`ODC-By v1.0`**.
   - **`FinePDFs-Edu (eng_Latn)`:** Longest docs (`3,240.8` tok/doc, `P99 = 30,095`), but **26.2%** school/admin PDFs (`T07`), only **2.0%** tech (`T06`), and SEO ebook spam in spot-checks — unsuitable as our sole standalone dataset.

---

## 3. Final Single-Dataset Selection: `Nemotron-ClimbMix (400B)`

Sticking to **one dataset for the entire training run** keeps Kestrel's training pipeline simple and reproducible. Based on our test-set-blind corpus audit combined with NVIDIA's CLIMB paper ([arXiv:2504.13161](https://arxiv.org/pdf/2504.13161)), **Kestrel selects `Nemotron-ClimbMix (400B)`** as its pretraining corpus:

- **Primary Pick (Selected): `Nemotron-ClimbMix (400B)`** — Highest scaling performance across 1B pretraining runs (`52.43%` across 12 benchmarks at 400B tokens, outperforming `DCLM-Baseline` and `FineWeb-Edu`), `60.6%` science + practical Q&A concentration in our BERT topic audit, and compact 400B-token footprint *(non-commercial `CC-BY-NC-4.0` upstream terms)*.
- **Commercially Permissive Runner-Up: `DCLM-Baseline 1.0`** (`CC-BY-4.0`) — Strongest commercially permissive track record (`48.9` vs. `41.9` CORE and `50.8` vs. `37.3` MMLU over `FineWeb-Edu` at 280B tokens), kept as backup if commercial licensing is required later.
