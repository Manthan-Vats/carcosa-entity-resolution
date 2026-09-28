# Backup and restore

Code, docs and logs live in this repo. Large files are attached to the `v1.0-final` release
(checksums in `SHA256SUMS.txt`).

| Asset | Contents | Restore to |
|---|---|---|
| `student_resource_dataset.zip` | original challenge data (train/test TSVs, README, validator) | unzip into `data/raw/` |
| `train_splits.parquet` | fold assignment f0-f4 + lockbox | `data/splits/` |
| `models.zip` | LightGBM models: v1, r002, r003, r004 (final), smoke; `translit_dict.json` | `models_r004/*` -> `models/`, `translit_dict.json` -> `data/interim/` |
| `outputs.zip` | test predictions: v1 (LB 0.902), r002 (0.927), r004 final (0.959) | `outputs/` |
| `carcosa_submission.zip` | the final submitted package | - |

## Restore

```
gh repo clone Manthan-Vats/carcosa-entity-resolution
cd carcosa-entity-resolution
gh release download v1.0-final
sha256sum -c SHA256SUMS.txt
unzip student_resource_dataset.zip -d data/raw -x "__MACOSX/*"
mkdir -p data/splits data/interim && mv train_splits.parquet data/splits/
unzip models.zip && mv models models_all && cp -r models_all/models_r004 models && cp models_all/translit_dict.json data/interim/
uv sync
uv run python scripts/convert_to_parquet.py
```

`data/interim/` (about 13 GB of parquet caches and blocking results) was not backed up; the
scripts rebuild it on first run (`run_dev.py` about 1 h, `run_test.py` per country).
