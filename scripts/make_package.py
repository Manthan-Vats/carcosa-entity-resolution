"""Assemble <team>_submission.zip: output/ TSVs, code/business_entity_resolution/, methodology doc."""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAM = "carcosa"
STAGE = ROOT / "package" / f"{TEAM}_submission"
CODE = STAGE / "code" / "business_entity_resolution"
PIPELINE = ["convert_to_parquet.py", "make_splits.py", "build_translit.py", "run_dev.py", "block_test.py", "run_test.py"]
ANALYSIS = ["blocking_recall.py", "blocking_union.py", "recall_depth.py", "maxdf_grid.py"]

if STAGE.exists():
    shutil.rmtree(STAGE)
(STAGE / "output").mkdir(parents=True)
for f in ("matching_results.tsv", "candidate_pairs.tsv"):
    shutil.copy2(ROOT / "outputs" / f, STAGE / "output" / f)

shutil.copytree(ROOT / "src" / "ber", CODE / "src" / "ber", ignore=shutil.ignore_patterns("__pycache__"))
for sub, names in (("scripts", PIPELINE), ("scripts/analysis", ANALYSIS)):
    (CODE / "src" / sub).mkdir(parents=True, exist_ok=True)
    for n in names:
        s = (ROOT / "scripts" / n).read_text(encoding="utf-8")
        # in the package the ber package sits in src/, next to src/scripts/
        up = 2 if sub.endswith("analysis") else 1
        s = s.replace('Path(__file__).resolve().parents[1] / "src"', f"Path(__file__).resolve().parents[{up}]")
        (CODE / "src" / sub / n).write_text(s, encoding="utf-8")
v = (ROOT / "scripts" / "validate.sh").read_text(encoding="utf-8").replace('$(dirname "$0")/..', '$(dirname "$0")/../..')
(CODE / "src" / "scripts" / "validate.sh").write_text(v, encoding="utf-8", newline="\n")
# learned artefacts at the same relative paths the code reads them from
for f in ("data/interim/translit_dict.json", "data/splits/train_splits.parquet", "models"):
    src, dst = ROOT / f, CODE / f
    dst.parent.mkdir(parents=True, exist_ok=True)
    (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
shutil.copy2(ROOT / "docs" / "package" / "README.md", CODE / "README.md")
req = subprocess.run(["uv", "export", "--no-hashes", "--no-dev", "--format", "requirements-txt"],
                     cwd=ROOT, capture_output=True, text=True, check=True).stdout
(CODE / "requirements.txt").write_text(re.sub(r"(?m)^\s*#.*\n", "", req), encoding="utf-8")
shutil.copy2(ROOT / "docs" / "package" / "Documentation_template.md", STAGE / "Documentation_template.md")

zpath = ROOT / "package" / f"{TEAM}_submission.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(STAGE.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(STAGE))
print(zpath, f"{zpath.stat().st_size / 1e6:.0f} MB", file=sys.stderr)
