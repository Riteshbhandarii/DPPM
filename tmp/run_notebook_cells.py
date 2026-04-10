from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 -u tmp/run_notebook_cells.py <notebook_path>")
        return 2

    notebook_path = Path(sys.argv[1]).resolve()
    repo_root = notebook_path.parents[2]

    with notebook_path.open() as f:
        notebook = json.load(f)

    ns = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        print(f"--- executing cell {index} ---", flush=True)
        code = compile(source, f"{notebook_path}:cell{index}", "exec")
        exec(code, ns)

    print(f"completed {notebook_path.relative_to(repo_root)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
