# Kestrel Pretraining Dataset Audit & Mixture Justification

> **Model Target:** **Kestrel** (350M–700M parameter modern decoder-only Transformer — RoPE, RMSNorm, SwiGLU, GQA/MQA, GPT-2 BPE or modern 50k–64k tokenizer) engineered to **outperform OpenAI's 1.5B GPT-2 XL** across all core GPT-2 benchmarks (**WikiText-103**, **LAMBADA**, **HellaSwag**, **ARC-Easy**) with 2–4× fewer parameters and zero benchmark contamination.

---

## 1. Candidate Pretraining Pool Comparison

To build a rigorous empirical foundation, we expand beyond the four core baseline corpora by incorporating two **2025–2026 compact-LM breakthrough corpora** (**Nemotron-ClimbMix** and **FinePDFs-Edu**) proven to outperform standalone FineWeb-Edu and DCLM-Baseline at sub-1B parameter scales, plus curated **stage-2 annealing pools** (**Dolmino-Mix 1124** and **SmolLM-Corpus**).

All six primary candidates are audited on equal-sized **3.5M–5.0M GPT-2 token slices** (`max_docs=5,000`, tokenized via `tiktoken` `gpt2`) using [`src/kestrel/data/diagnostics.py`](../src/kestrel/data/diagnostics.py) and [`notebooks/01_dataset_audit.ipynb`](../notebooks/01_dataset_audit.ipynb).

| Dataset | Hugging Face Repo ID & Streaming Config | License & Access | Scale | Noise & Structural Profile | GPT-2 / Benchmark Role | Recommended Role in Kestrel |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FineWeb-Edu (10BT)** *(Baseline)* | `HuggingFaceFW/fineweb-edu` (`name="sample-10BT"`, `split="train"`) | **ODC-By v1.0** (Open, ungated, commercial OK) | ~1.3T full / 10BT sample | Low boilerplate (`0.78%` cross-doc, `5.06%` Gopher rep); slight stylistic narrowing toward academic/tutorial phrasing | Very strong **ARC-Easy** & **WikiText-103** alignment; moderate JSD drift from casual **LAMBADA** narrative prose | **Core Pillar (Base + Annealing)** — factual density & STEM exposition |
| **DCLM-Baseline 1.0** *(Baseline)* | `mlfoundations/dclm-baseline-1.0-parquet` *(fallback: raw `.jsonl.zst` at `mlfoundations/dclm-baseline-1.0`)* | **CC-BY-4.0** (Open, ungated, commercial OK) | ~3.8T tokens | FastText quality-filtered web; lowest within-doc repetition (`1.21%` Gopher); occasional web-navigation boilerplate (`3.15%` cross-doc) | Closest natural-web distribution to **OpenWebText** (`0.061` JSD), **HellaSwag**, and **LAMBADA**; prevents perplexity drift | **Core Pillar (Base + Annealing Anchor)** — linguistic diversity & commonsense fluency |
| **Nemotron-CC v2.1 (HQ)** *(Baseline)* | `nvidia/Nemotron-CC-v2.1` (`data_dir="High-Quality"`) *(fallback: `Nemotron-CC-v2` / `Nemotron-Pretraining-Dataset-sample`)* | **NVIDIA Data Agreement for Model Training** (Gated; requires HF login & agreement) | ~2.5T+ new tokens (6.3T+ v1/v2 family) | Lynx + LLM extraction & synthetic Q&A/rephrasing preserves LaTeX math, tables, and clean code structure | Strong multi-domain bridge across **ARC-Easy**, **HellaSwag**, and technical **WikiText-103** | **Core Pillar (Base)** — high-signal structured web & synthetic diversity |
| **Nemotron-ClimbMix (400B)** *(New 2025–2026 #1)* | `karpathy/climbmix-400b-shuffle` *(upstream: `nvidia/Nemotron-ClimbMix`; text mirror: `OptimalScale/ClimbMix`)* | **CC-BY-NC-4.0** upstream *(MIT on Karpathy mirror — verify NC terms before commercial release)* | 400B tokens | CLIMB semantic-cluster-bootstrapped mix tuned via **62M & 350M proxy models**; ultra-high sample efficiency (~33% faster GPT-2-scale convergence) | Beat FineWeb-Edu, DCLM, and Llama-3.2-1B (+2.0% at 1B) on reasoning + language modeling under matched token budgets | **Research-Track Primary Backbone** *(omitted from strictly commercial track due to upstream NC license)* |
| **FinePDFs-Edu (`eng_Latn`)** *(New 2025–2026 #2)* | `HuggingFaceFW/finepdfs-edu` (`name="eng_Latn"`, `split="train"`) | **ODC-By v1.0** + Common Crawl ToU (Open, ungated, commercial OK) | ~350B tokens (69 langs; ~3T unfiltered FinePDFs) | Long-form textbooks, papers, manuals, reports (~3,240 tok/doc); watch repeated PDF headers/tables caught by Gopher repetition (`21.85%`) | Proven winner in Hugging Face's *1B Token Challenge* for <1B models when blended with DCLM + FineWeb-Edu; massive boost to **ARC-Easy** & long-range coherence | **Core Pillar (Base + Annealing)** — deep expository structure & long-range context |
| **OpenWebText** *(Control)* | `Skylion007/openwebtext` *(Parquet fallback: `refs/convert/parquet`)* | **Public Domain / CC0** (Community WebText recreation) | ~9B tokens (~38 GB text) | Unfiltered 2019 Reddit outbound links (≥3 karma); noticeably higher Gopher repetition & boilerplate than modern curated corpora | **Empirical GPT-2 Distribution Anchor** — reference point for measuring domain JSD vs GPT-2 XL | **Diagnostic Control Only** — *not* mixed into training |
| **Dolmino-Mix 1124 & SmolLM-Corpus** *(Annealing Pool)* | `allenai/dolmino-mix-1124` (`wiki`, `pes2o`, `math`, `stackexchange`, `flan`) & `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2`, `python-edu`) | **ODC-By v1.0** (Open, ungated) | ~100B–250B+ tokens across subsets | Ultra-clean encyclopedic, synthetic textbook, math, and educational code subsets designed specifically for stage-2 LR decay | Sharpens **WikiText-103** (encyclopedic style), **ARC-Easy** (science/textbook logic), and compositional reasoning during final 10% decay | **Stage-2 Annealing Pool (Final 10% of tokens)** |

---

## 2. Diagnostic Guardrails (Empirical Audit Protocol)

Before committing full GPU compute to any candidate slice in the mixture, run [`notebooks/01_dataset_audit.ipynb`](../notebooks/01_dataset_audit.ipynb) on Kaggle. To avoid opening-shard prefix bias, `stream_matched_slice()` samples evenly across **8 randomly selected remote shards** with a per-shard shuffle buffer (and `stream_aggregate_audit()` is available for constant-RAM whole-dataset sweeps). Every active data source is checked against three non-overlapping quantitative gates implemented in [`src/kestrel/data/diagnostics.py`](../src/kestrel/data/diagnostics.py):

1. **Gopher Repetition & Cross-Doc 7-Gram Boilerplate:**
   - Apply length-normalized **Gopher/FineWeb within-document n-gram repetition filters** (`top_{2,3,4}gram` and `dup_{5..10}gram`). Any source exceeding **10% Gopher flag rate** requires a lightweight pre-tokenization filter pass before training.
   - Run **cross-document 7-gram boilerplate detection** (`audit_cross_doc_boilerplate`) to surface recurring cookie banners, footers, and scraped headers that appear once per page.
2. **Exploratory Data Analysis & BERT Semantic Topic Composition:**
   - Inspect document length distributions (GPT-2 tokens/doc), structural quality indicators (digit/symbol/uppercase ratios), and **BERT (`all-MiniLM-L6-v2`) semantic topic clusters** automatically labeled with `google/flan-t5-small` to verify balanced coverage across STEM, narrative, technical, and general-world topics.
3. **GPT-2 & Benchmark Distributional Proximity (Unigram/Bigram JSD):**
   - Measure **unigram and bigram Jensen-Shannon Divergence** (base-2, in $[0, 1]$ bits) and **Top-10k vocabulary Jaccard** against both **OpenWebText** and the four evaluation benchmarks (**HellaSwag**, **ARC-Easy**, **LAMBADA**, **WikiText-103**).
   - Healthy mixtures minimize JSD jointly across *both* narrative/commonsense benchmarks (OpenWebText, LAMBADA, HellaSwag) and encyclopedic/scientific benchmarks (WikiText-103, ARC-Easy).
4. **13-Gram Exact-Overlap Decontamination Gate:**
   - Compute normalized alphanumeric **13-gram overlap** against all four benchmark evaluation splits.
   - **Hard policy:** Any document matching an evaluation 13-gram is **removed** from the training stream prior to packing. Any candidate dataset exhibiting **> 0.5% eval-example 13-gram overlap** on a multi-shard probe must undergo full-corpus 13-gram decontamination (and be capped or audited for scraped benchmark mirrors).

---

## 3. Recommended Kestrel Training Mixture Ratios

A 350M–700M parameter model trained in the modern inference-efficient regime (**100B–250B tokens**, i.e., ~150–350 tokens/param, well past Chinchilla's 20× compute-optimal point) is **capacity-constrained** rather than token-starved. Every noisy or redundant token consumes precious representational capacity that a 1.5B model like GPT-2 XL could absorb.

We therefore use a **two-stage WSD (Warmup–Stable–Decay) curriculum**:
- **Stage 1 — Base Pretraining (First 90% of tokens):** Constant/gentle-decay learning rate over a high-diversity, high-density multi-source foundation.
- **Stage 2 — High-Density Annealing (Final 10% of tokens):** Linear/cosine decay to 0 LR over ultra-clean educational, encyclopedic, synthetic textbook, and reasoning data—anchored by a 15% general-web slice so perplexity never regresses.

### Track A: Primary Commercial-Safe Mixture *(Recommended Default — 100% ODC-By / CC-BY-4.0 / NVIDIA Model Training Agreement)*

#### Stage 1: Base Pretraining Mix (90% of Total Token Budget)

| Component | HF Source | Ratio | Primary Capability Contribution |
| :--- | :--- | :---: | :--- |
| **DCLM-Baseline 1.0** | `mlfoundations/dclm-baseline-1.0-parquet` | **32%** | Broadest natural web prose, everyday commonsense transitions (**HellaSwag**), narrative & colloquial fluency (**LAMBADA**), low JSD to OpenWebText |
| **FinePDFs-Edu (`eng_Latn`)** | `HuggingFaceFW/finepdfs-edu` (`eng_Latn`) | **25%** | Long-form structured documents (textbooks, technical manuals, scientific papers); multi-paragraph coherence (**WikiText-103**) & structured reasoning (**ARC-Easy**) |
| **FineWeb-Edu** | `HuggingFaceFW/fineweb-edu` (`default` / deduped) | **23%** | High-density grade-school to college STEM & humanities explanations; steepens scaling curve on **ARC-Easy** |
| **Nemotron-CC v2.1 (HQ)** | `nvidia/Nemotron-CC-v2.1` (`High-Quality`) | **15%** | Clean structured web + synthetically rephrased Q&A & preserved math/tables; bridges web fluency with textbook rigor |
| **Educational Code & Math Scaffold** | `HuggingFaceTB/smollm-corpus` (`python-edu`) + `HuggingFaceTB/finemath` (`finemath-4plus`) | **5%** | Algorithmic and quantitative reasoning scaffold; boosts logical deduction on **ARC-Easy** without diluting prose perplexity |
| **Total (Stage 1)** | | **100%** | *(If `Nemotron-CC v2.1` gated access is unavailable, reallocate its 15% as +8% `FineWeb-Edu`, +4% `DCLM-Baseline`, +3% `FinePDFs-Edu`.)* |

#### Stage 2: Final 10% Annealing Mix (LR Decay to 0)

| Component | HF Source | Ratio | Primary Capability Contribution |
| :--- | :--- | :---: | :--- |
| **FineWeb-Edu (High-Score Slice, ≥4)** | `HuggingFaceFW/fineweb-edu` (filtered `int_score >= 4`) | **25%** | Concentrates crisp explanatory prose and scientific world knowledge during final weight settling |
| **FinePDFs-Edu (`eng_Latn`, top quality)** | `HuggingFaceFW/finepdfs-edu` (`eng_Latn`) | **20%** | Reinforces long-range discourse coherence and formal expository transitions |
| **Dolmino-Mix 1124 (`wiki` + `pes2o` + `math` + `stackexchange`)** | `allenai/dolmino-mix-1124` (12% `wiki`, 8% `pes2o`, 5% `math`/`stackexchange`) | **25%** | Clean encyclopedic prose directly lowers **WikiText-103** perplexity (decontaminated against WT-103 test) while `pes2o`/`math` sharpen **ARC-Easy** |
| **Cosmopedia v2 (Synthetic Textbooks)** | `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2`) | **15%** | High-signal pedagogical explanations & multi-step reasoning chains optimized for sub-1B parameter capacity |
| **DCLM-Baseline 1.0 (Linguistic Anchor)** | `mlfoundations/dclm-baseline-1.0-parquet` | **15%** | Critical anti-forgetting anchor: prevents annealing from over-specializing to academic tone and keeps **LAMBADA** & **HellaSwag** perplexity razor-sharp |
| **Total (Stage 2)** | | **100%** | |

---

### Track B: Accelerated Research-Only Variant *(Incorporating `Nemotron-ClimbMix`, CC-BY-NC-4.0 Upstream)*

For non-commercial research benchmarks or rapid GPT-2-scale speedruns where NVIDIA's upstream **CC-BY-NC-4.0** research terms are acceptable, `Nemotron-ClimbMix` (`karpathy/climbmix-400b-shuffle`) replaces part of the web-only foundation because its CLIMB cluster weights were explicitly bootstrapped on 62M–350M proxy models:

- **Stage 1 (90%):** **45%** `Nemotron-ClimbMix` + **22%** `FinePDFs-Edu (eng_Latn)` + **18%** `DCLM-Baseline 1.0` + **10%** `FineWeb-Edu` + **5%** `Python-Edu / FineMath`.
- **Stage 2 (10%):** Identical to Track A Stage 2, substituting `DCLM-Baseline 1.0` (15%) with **10% `DCLM-Baseline` + 5% `Nemotron-ClimbMix`**.

---

## 4. Why This Mixture Beats GPT-2 XL (1.5B) on Both Perplexity and Reasoning — Without Contamination

### 4.1 Why GPT-2 XL (1.5B) Is Beatable at 350M–700M Parameters
OpenAI's GPT-2 XL (1.5B parameters) was trained on ~40 GB (~9–10B tokens) of **WebText** (roughly 10 epochs / ~100B total seen tokens with massive repetition, pre-norm/LayerNorm, learned absolute positional embeddings, and zero classifier-based quality filtering). Modern sub-1B models outperform GPT-2 XL through two compounding multipliers:
1. **10–25× Higher Unique Token-to-Parameter Density:** Training Kestrel on 100B–250B *unique, deduplicated* tokens eliminates the epoch-memorization ceiling that bottlenecked GPT-2 XL.
2. **Signal-to-Noise Density per Token:** Filtering out repeated n-gram boilerplate (Gopher repetition & cross-doc 7-gram headers/footers) frees Kestrel's 350M–700M parameters to learn coherent world models instead of memorizing HTML menus and template loops.

### 4.2 Resolving the Perplexity vs. Reasoning Trade-Off
Single-dataset pretraining recipes consistently fail one half of the GPT-2 benchmark suite:

- **Failure Mode 1 — Pure Educational Corpora (`100% FineWeb-Edu` or `100% Cosmopedia`):**
  - *Result:* Crushes **ARC-Easy** (+10–15 pts over GPT-2 XL) because every document resembles a textbook.
  - *Failure:* Exhibits **higher Jensen-Shannon Divergence** against **OpenWebText**, **LAMBADA** (book-length narrative final-word prediction), and **HellaSwag** (everyday physical/how-to commonsense). A sub-1B model trained exclusively on academic prose loses conversational, narrative, and idiomatic transition probabilities, causing **LAMBADA** accuracy and **HellaSwag** likelihoods to underperform.
- **Failure Mode 2 — Pure Filtered Web Crawl (`100% DCLM-Baseline` or `OpenWebText`):**
  - *Result:* Achieves strong **LAMBADA** and **HellaSwag** fluency and minimal JSD to OpenWebText.
  - *Failure:* At 350M–700M parameters, raw web text lacks sufficient density of explicit causal explanations and structured scientific definitions to beat a 1.5B model decisively on **ARC-Easy** and formal **WikiText-103** prose.

**How Kestrel's Mixture Solves Both Simultaneously:**
1. **Perplexity Dominance (WikiText-103 & LAMBADA):**
   - **WikiText-103** rewards long-range encyclopedic coherence, clean named-entity continuity, and formal expository transitions. **FinePDFs-Edu (25% base / 20% anneal)** supplies multi-page structured documents (much longer than web pages), while **Dolmino `wiki` + `pes2o`** during the final 10% annealing phase calibrates encyclopedic register and token probabilities cleanly.
   - **LAMBADA** rewards broad narrative, literary, and discourse-level coreference tracking over 100+ tokens. Keeping **DCLM-Baseline at 32% in Stage 1 and 15% in Stage 2** preserves broad everyday English coverage and low JSD to general narrative prose, preventing the "textbook overfitting" that hurts LAMBADA on educational-only models.
2. **Reasoning Dominance (HellaSwag & ARC-Easy):**
   - **HellaSwag** evaluates everyday physical, procedural, and commonsense continuation (WikiHow / ActivityNet style). The combination of **DCLM-Baseline (32%) + Nemotron-CC HQ (15%)** provides rich procedural and real-world event distributions, while **FinePDFs-Edu** manuals and **Cosmopedia-v2** reinforce step-by-step causal chains.
   - **ARC-Easy** evaluates grade-school science knowledge and causal deduction. **FineWeb-Edu (23%→25%) + FinePDFs-Edu (25%→20%) + Nemotron-CC HQ (15%) + Code/Math (5%)** saturate the model with clean scientific explanations, quantitative relationships, and textbook Q&A structures—allowing a 350M–700M model to surpass GPT-2 XL (1.5B) on ARC-Easy by a wide margin.

### 4.3 Healthy Domain Proximity vs. Benchmark Contamination
A core design principle of Kestrel's data audit is separating **healthy domain proximity** from **test-set leakage**:
- **Healthy Domain Proximity:** Low unigram and bigram **JSD** against WikiText-103, LAMBADA, HellaSwag, and ARC-Easy mean Kestrel sees the same *register, vocabulary, and conceptual primitives* used in English prose and science—without ever seeing the test items themselves.
- **Strict Decontamination:** Every dataset in Kestrel's pipeline is checked with our **13-gram exact-overlap detector** (`audit_contamination_13gram`) against all four evaluation splits, and production data-packing drops any training document sharing a normalized 13-gram with any benchmark test/validation example (especially critical for Wikipedia-derived subsets like `Dolmino wiki` vs. `WikiText-103 test`, and web scrapes vs. `HellaSwag` WikiHow articles).
- **Clean Control Boundary:** `OpenWebText` is used exclusively as an empirical baseline in the audit notebook to measure distributional distance from GPT-2's training set, ensuring Kestrel's gains reflect genuinely superior 2025–2026 data curation rather than replicating GPT-2's WebText distribution.
