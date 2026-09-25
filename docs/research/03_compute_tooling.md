# Research 03: Compute & Tooling (research notes, 2026-09-25)

> Status: raw research findings. Items marked UNVERIFIED are not trusted. Nothing here is adopted until we discuss and decide it (see docs/DECISIONS.md once created).

## Free compute
| Platform | GPU | Quota | Runs in background? | Notes |
|---|---|---|---|---|
| Kaggle | P100 16GB / T4x2 | ~30 GPU-h/week per account, 12h per session | Yes (`kaggle kernels push`) | /kaggle/working ~20GB; private datasets up to 200GB; internet off by default |
| Colab Free | T4 (when available) | ≤12h, ~90 min idle disconnect | No | Interactive only; official VS Code extension exists (Nov 2025); tunnels break the ToS |
| Lightning AI | T4+ | 15 credits/month (figures partly UNVERIFIED) | Yes | needs phone verification |
| Modal Starter | T4–H100 | $30/month credit | Yes | card policy UNVERIFIED |
| SageMaker Studio Lab | ? | 4 GPU-h/day | No | |
3 teammates ≈ 90 Kaggle GPU-h/week.

## Proposed hybrid
- Local: CPU work (pandas, blocking, features, LightGBM) + small GPU jobs (≤~150M params fine-tune) on the RTX 3050 6GB.
- Kaggle: heavy fine-tuning, driven by CLI (`kaggle datasets create/version`, `kernels push/status/output`). Docs: https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md
- Native Windows + uv with an explicit PyTorch CUDA index (https://docs.astral.sh/uv/guides/integration/pytorch/). Skip WSL2: its disk image grows and we only have ~39GB free.
- Set HF_HOME to a chosen folder and enable Windows Developer Mode so HF cache symlinks work (otherwise model files are stored twice).
- Convert TSVs to Parquet once.

## Tooling verdicts
- Kaggle CLI: USE. Official Kaggle MCP (kaggle.com/mcp): MAYBE. Community Kaggle MCPs: SKIP.
- jupyter-mcp-server: MAYBE (only if we use notebooks). HF skills plugin: MAYBE.
- rtk: `winget install rtk-ai.rtk` then `rtk init -g` (https://github.com/rtk-ai/rtk/blob/master/INSTALL.md). Whether its hook works on native Windows is UNVERIFIED.
- headroom MCP: MAYBE. Results log: a shared `results.csv` (USE); MLflow optional; W&B MAYBE; DVC SKIP.
- Git: one repo, short-lived experiment branches, CV folds fixed once and saved to a file, data/models kept out of git.

## Pitfalls
Colab idle disconnects · Kaggle files lost after 12h unless in /kaggle/working · uv installs CPU-only torch unless the CUDA index is set · 6GB VRAM too small for >1B LoRA · pin identical versions for local and Kaggle · keep big outputs out of the terminal.
