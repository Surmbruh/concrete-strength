"""Convert Colab-exported .py to .ipynb with proper markdown cells.

Recognizes triple-quoted strings (\"\"\"...\"\"\") as markdown cells
and everything between them as code cells.

Usage:
    python convert_to_notebook.py new_final_report.py
"""
import json
import re
import sys


def py_to_notebook(py_path: str, ipynb_path: str | None = None):
    if ipynb_path is None:
        ipynb_path = py_path.replace(".py", ".ipynb")

    with open(py_path, "r", encoding="utf-8") as f:
        content = f.read()

    cells = []
    # Split by triple-quoted blocks
    # Pattern: content outside """ is code, content inside """ is markdown
    parts = re.split(r'"""', content)

    for i, part in enumerate(parts):
        text = part.strip()
        if not text:
            continue

        if i % 2 == 0:
            # Code block — split by "# %%"  markers if present
            code_chunks = re.split(r'^# %%.*$', text, flags=re.MULTILINE)
            for chunk in code_chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                # Skip the encoding header
                if chunk.startswith("# -*- coding"):
                    continue
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [line + "\n" for line in chunk.split("\n")]
                })
        else:
            # Markdown block
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in text.split("\n")]
            })

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells
    }

    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"Created {ipynb_path} ({len(cells)} cells: "
          f"{sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown, "
          f"{sum(1 for c in cells if c['cell_type'] == 'code')} code)")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "new_final_report.py"
    py_to_notebook(src)
