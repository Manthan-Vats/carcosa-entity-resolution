"""Convert a `# %%` cell script to a notebook: python scripts/study/py_to_ipynb.py notebooks/predictive_modeling.py"""
import re
import sys
from pathlib import Path

import nbformat

src = Path(sys.argv[1])
cells = []
for block in re.split(r"^# %%", src.read_text(encoding="utf-8"), flags=re.M)[1:]:
    head, _, body = block.partition("\n")
    if head.strip() == "[markdown]":
        text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.strip().splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip()))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(nb, src.with_suffix(".ipynb"))
print(f"{len(cells)} cells -> {src.with_suffix('.ipynb')}")
