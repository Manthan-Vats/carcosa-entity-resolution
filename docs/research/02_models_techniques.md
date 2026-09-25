# Research 02: Pretrained Models & Training Techniques (research notes, 2026-09-25)

> Raw research findings. Licenses marked "verified" were read from the HF card or API. Nothing here is adopted until we decide it. Parameter counts marked ~ are approximate or UNVERIFIED.

## Embedding models (bi-encoders)
| Model | Params | License | 6GB? | URL |
|---|---|---|---|---|
| intfloat/multilingual-e5-small | ~118M | MIT ✓ | yes | https://huggingface.co/intfloat/multilingual-e5-small |
| intfloat/multilingual-e5-base | ~278M | MIT ✓ | yes | https://huggingface.co/intfloat/multilingual-e5-base |
| intfloat/multilingual-e5-large-instruct | ~560M | MIT ✓ | infer yes; train with fp16 + CachedMNRL | https://huggingface.co/intfloat/multilingual-e5-large-instruct |
| BAAI/bge-m3 | ~568M | MIT ✓ | same as above; has dense, sparse and multi-vector outputs | https://huggingface.co/BAAI/bge-m3 |
| Alibaba-NLP/gte-multilingual-base | 305M | Apache ✓ | yes; needs trust_remote_code | https://huggingface.co/Alibaba-NLP/gte-multilingual-base |
| Snowflake arctic-embed-m/l-v2.0 | 113M/303M non-embedding | Apache ✓ | yes/tight | https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0 |
| nomic-embed-text-v2-moe | 475M (305M active) | Apache ✓ | yes | https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe |
| paraphrase-multilingual-MiniLM-L12-v2 / mpnet-base-v2 | 0.1B/0.3B | Apache ✓ | yes | sentence-transformers |
| sentence-transformers/LaBSE | ~0.5B | Apache ✓ | yes | https://huggingface.co/sentence-transformers/LaBSE |
| Qwen3-Embedding-0.6B / 8B | 0.6B/8B | Apache ✓ (4B not checked) | 0.6B yes | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B |
| EXCLUDED: jina-embeddings-v3 (CC-BY-NC), embeddinggemma (Gemma license) | | | | |

## Rerankers (cross-encoders)
| Model | Params | License | URL |
|---|---|---|---|
| BAAI/bge-reranker-v2-m3 | 0.6B | Apache ✓ | https://huggingface.co/BAAI/bge-reranker-v2-m3 |
| Alibaba-NLP/gte-multilingual-reranker-base | 306M | Apache ✓ | https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base |
| Qwen3-Reranker-0.6B / 4B | 0.6B/4B | Apache ✓ | https://huggingface.co/Qwen/Qwen3-Reranker-0.6B |
| mixedbread-ai/mxbai-rerank-base-v2 | 0.5B | Apache ✓ | https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2 |

## Small LLMs (≤8B)
OK: Qwen3-1.7B, Qwen3-4B-Instruct-2507, Qwen2.5-1.5B (Apache ✓); Phi-4-mini (MIT ✓, has French, no Hindi); Mistral-7B-v0.3, SmolLM3-3B, granite-3.3-2b, OLMo-2-7B (Apache ✓).
EXCLUDED: **Qwen2.5-3B (Qwen Research license, non-commercial)**, Jellyfish-8B (CC-BY-NC, built on Llama-3), Llama and Gemma families.
- AnyMatch (GPT-2 fine-tuned for matching, https://arxiv.org/abs/2409.04073): a method only; repo license UNVERIFIED.
- Peeters et al. 2024 (https://arxiv.org/abs/2409.08185): fine-tuning an LLM for matching **hurts cross-domain transfer**. This is a warning for France.

## Technique verdicts
- USE: sentence-transformers CachedMNRL (large in-batch negatives on 6GB); `util.mine_hard_negatives` (https://sbert.net/docs/package_reference/util/hard_negatives.html); cross-encoder fine-tuning; GBDT stacker; calibration + F0.5 threshold; leave-one-country-out validation; accent folding; char n-grams; cross-language legal-suffix stripping.
- MAYBE: model soups (only for fine-tunes from the same init; improves out-of-distribution robustness, https://arxiv.org/abs/2203.05482); QLoRA LLM matcher used only on borderline pairs, trained on Kaggle.
- SKIP: SetFit; Unsloth on native Windows.

## Windows notes
- bitsandbytes: has Windows wheels (cu130 compatibility UNVERIFIED).
- flash-attn: no official Windows wheels, so use SDPA.
- faiss-gpu: Linux-only, so use faiss-cpu or a torch top-k.
- LightGBM: CUDA is Linux-only; on Windows the GPU build is OpenCL, and CPU is fine.
- XGBoost and CatBoost: GPU works on Windows.

## Top-5
1. Fine-tune e5-base or gte-multilingual-base (CachedMNRL + hard negatives) for retrieval, plus a TF-IDF char n-gram or bge-m3 sparse recall channel.
2. Fine-tune the bge-reranker-v2-m3 cross-encoder for precision (fall back to the gte reranker if memory is tight).
3. GBDT stacker on model scores + string features, then isotonic calibration, then F0.5 thresholding.
4. Leave-one-country-out validation plus augmentation that imitates France; soup checkpoints.
5. Optional: a Qwen3 reranker or QLoRA Qwen3-4B only on borderline pairs.
