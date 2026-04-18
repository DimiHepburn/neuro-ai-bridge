"""
py_to_ipynb.py
==============

Convert the repo's `% %`-cell Python files into proper .ipynb
notebooks that render nicely on GitHub and can be executed in
Jupyter.

The converter understands the two cell markers used in this repo:

    # %%                  -> new code cell
    # %% [markdown]       -> new markdown cell (comment lines below
                             are unwrapped as markdown)

No external dependencies — uses only Python stdlib + nbformat.
(Install with: ``pip install nbformat``.)

Usage
-----

From the repo root:

    # Convert every notebooks/*.py -> notebooks/*.ipynb
    python scripts/py_to_ipynb.py

    # Convert a single file
    python scripts/py_to_ipynb.py notebooks/01_hebbian_learning.py

    # Convert into a different output directory
    python scripts/py_to_ipynb.py notebooks/ --out build/notebooks

Author: Dimitri Romanov
Project: neuro-ai-bridge
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable, List, Tuple

try:
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
except ImportError as e:  # pragma: no cover
    print("This script needs `nbformat`.  Install it with:", file=sys.stderr)
    print("    pip install nbformat", file=sys.stderr)
    raise SystemExit(1) from e


CELL_MARK = "# %%"
MARKDOWN_MARK = "# %% [markdown]"


def _parse_cells(source: str) -> List[Tuple[str, str]]:
    """
    Split a source file into (cell_type, body) tuples.

    Rules
    -----
    * Lines that start with ``# %% [markdown]`` open a markdown cell.
    * Lines that start with ``# %%`` (no `[markdown]`) open a code cell.
    * Everything before the first marker is treated as a leading code cell
      (useful for imports / headers).
    * Inside markdown cells, leading ``# `` / ``#`` on each line is
      stripped so the comments render as real markdown.
    """
    lines = source.splitlines()
    cells: List[Tuple[str, List[str]]] = []
    current_type = "code"
    current_body: List[str] = []

    def flush():
        if current_body and any(l.strip() for l in current_body):
            cells.append((current_type, current_body.copy()))
        current_body.clear()

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(MARKDOWN_MARK):
            flush()
            current_type = "markdown"
            continue
        if stripped.startswith(CELL_MARK):
            flush()
            current_type = "code"
            continue
        current_body.append(line)

    flush()

    # Convert markdown cell bodies: strip leading `# ` / `#` per line.
    parsed: List[Tuple[str, str]] = []
    for ctype, body in cells:
        if ctype == "markdown":
            md_lines = []
            for l in body:
                s = l.lstrip()
                if s.startswith("# "):
                    md_lines.append(l[l.index("# ") + 2 :])
                elif s.startswith("#"):
                    md_lines.append(l[l.index("#") + 1 :])
                else:
                    md_lines.append(l)
            parsed.append(("markdown", "\n".join(md_lines).strip("\n")))
        else:
            parsed.append(("code", "\n".join(body).strip("\n")))

    return parsed


def convert_file(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Convert a single .py cell-script to an .ipynb notebook."""
    text = src.read_text(encoding="utf-8")
    cells = _parse_cells(text)

    nb = new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    for ctype, body in cells:
        if not body.strip():
            continue
        if ctype == "markdown":
            nb["cells"].append(new_markdown_cell(body))
        else:
            nb["cells"].append(new_code_cell(body))

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"  {src}  ->  {dst}")


def iter_targets(paths: Iterable[str]) -> List[pathlib.Path]:
    """Expand a list of file / directory arguments into .py files."""
    targets: List[pathlib.Path] = []
    for p in paths:
        path = pathlib.Path(p)
        if path.is_dir():
            targets.extend(sorted(path.glob("*.py")))
        elif path.suffix == ".py":
            targets.append(path)
        else:
            print(f"Skipping {path} (not a .py file or directory)",
                  file=sys.stderr)
    return targets


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert `# %%`-cell .py files into executable .ipynb "
            "notebooks."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["notebooks"],
        help=(
            "Files or directories to convert. Defaults to 'notebooks'."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output directory. Defaults to writing .ipynb files next to "
            "the corresponding .py files."
        ),
    )
    args = parser.parse_args(argv)

    targets = iter_targets(args.paths)
    if not targets:
        print("No .py files found to convert.", file=sys.stderr)
        return 1

    out_dir = pathlib.Path(args.out) if args.out else None
    print(f"Converting {len(targets)} file(s):")
    for src in targets:
        dst = (
            out_dir / (src.stem + ".ipynb")
            if out_dir is not None
            else src.with_suffix(".ipynb")
        )
        convert_file(src, dst)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
