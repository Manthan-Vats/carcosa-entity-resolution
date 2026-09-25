# Research 04: Data-specific problems, papers & tools (research notes, 2026-09-25)

> Raw research findings. Not adopted until decided.

## Devanagari ↔ Latin
- IndicXlit (MIT, ~11M, both directions): uses fairseq, which **fails to build on Windows**; the py3.12 fix PR #48 is unmerged. MAYBE, Kaggle/Linux only, run on unique Devanagari tokens only. https://github.com/AI4Bharat/IndicXlit
- indic_transliteration (MIT, rule-based, no handling of English loanwords). Aksharamukha is **GPL-3.0 → SKIP**.
- anyascii (ISC, fast): drops the inherent vowel ("महासमुंद"→"mhasmumd"). Use as a baseline plus consonant-skeleton matching.
- Embeddings: LaBSE (Apache), bge-m3 (MIT), e5 (MIT), **bhasha-embed-v0** (Apache, 0.2B, MuRIL-based, trained on Hindi / English / romanized Hindi, https://huggingface.co/AkshitaS/bhasha-embed-v0). RomanSetu: romanized text aligns with English better than native script does (https://arxiv.org/html/2401.14280v2). No benchmark on Devanagari names found (UNVERIFIED).
- **Idea: learn a Devanagari→Latin token dictionary from our own matched train pairs** (alignment by co-occurrence), which also covers loanwords (मार्केटिंग→marketing).

## Noise generators
Febrl dsgen, GeCo, Gecko (MIT): none matches our noise types (script swap, legal form moved to the front, domain used as name). The noise is likely custom → build our own augmenter from the noise types we observe.
⚠️ Public repo found: https://github.com/Amazon-ML-Challenge-2026/business-entity-resolution. Not known whether it's official or a participant's. **NOT inspected, pending a team decision.**

## Scale
- Sparkly: top-k BM25 on Lucene; 10M tuples in <100 min on 10 nodes. Use the idea, not the stack.
- sparse_dot_topn (Apache, Windows wheels, OpenMP): USE.
- BlockingPy (MIT, ANN blocking; its GPU build needs Linux): MAYBE.
- Splink ("1M records ~1 min" on DuckDB): MAYBE, for Fellegi-Sunter features.

## Company names / domain shift
- cleanco's coverage of prefix-position forms and Pvt/SARL is UNVERIFIED → mine our own list of legal-form tokens from train, position-independent.
- 2026 study (1,215 runs, https://arxiv.org/abs/2607.24688): cross-encoders > bi-encoders; generative matchers help mainly under distribution shift; bigger models lean more on shortcuts.
- AnyMatch: GPT-2 fine-tuned, F1 within 4.4% of GPT-4. Unicorn has no license → SKIP.

## F0.5 decision
Under independence, the optimal prediction = top-k by probability for some k∈{0..n} (Ye et al. 2012, https://arxiv.org/pdf/1206.4625). With n≈10–30 candidates, brute force over k with expected F0.5 is trivial, and k=0 is worth P(no match) (the PS confirms empty-vs-empty = 1.0). Exact without independence: Dembczyński GFM, O(n³).

## Soup
Apache; PowerShell install commands shown, so Windows works (not explicitly stated). Layer streaming is documented: Llama-3.1-8B NF4 at 119.6 tok/s, 3.32 GB peak on an RTX 3050 4GB (BETA, LoRA only, seq 512). Rerankers/cross-encoders are not documented. MAYBE/SKIP.

## Top-5
1. Devanagari→Latin token dictionary learned from train matches + anyascii fallback.
2. Canonical name key (junk/accents/domains/legal forms removed) for blocking + features.
3. Per-country top-k char-3gram TF-IDF blocking (sparse_dot_topn) unioned with key blocks; measure recall@k.
4. Expected-F0.5 top-k selection (k≥0) from calibrated probabilities; candidate-set-size features.
5. Leave-one-country-out validation; language-agnostic GBDT features; optional ≤600M cross-encoder.
