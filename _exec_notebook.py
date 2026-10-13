"""
Execute Assignment02.ipynb code cells and SAVE outputs into the .ipynb.
Usage:
  .\\.venv\\Scripts\\python.exe _exec_notebook.py --upto 25
  .\\.venv\\Scripts\\python.exe _exec_notebook.py --from 25 --upto 33
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / "Assignment02.ipynb"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--upto", type=int, required=True, help="Last cell index to execute (inclusive)")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    import copy

    nb = nbformat.read(NB_PATH, as_version=4)

    # Deep-copy prefix so we never mutate sources in the real notebook while executing
    run_nb = nbformat.v4.new_notebook(metadata=copy.deepcopy(nb.metadata))
    run_nb.cells = [copy.deepcopy(c) for c in nb.cells[: args.upto + 1]]

    # Clear outputs in the execution copy
    for cell in run_nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    # Replace shell-pip cell with a no-op ONLY in the execution copy
    for cell in run_nb.cells:
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if cell.cell_type == "code" and "!pip install" in src:
            cell.source = (
                "# (auto-exec) local env already has packages — pip skipped\n"
                "print('pip skipped (local venv)')\n"
            )

    # Do not execute cells before --from on this pass if restarting mid-notebook:
    # For mid-run we still need earlier state, so always execute from 0..upto unless
    # user passes --from with assumption prior state exists (not supported). Always 0..upto.
    # Prefer the GPU venv kernel we registered; fall back to python3
    kernel = "ann-dl-asm2"
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        specs = KernelSpecManager().find_kernel_specs()
        if kernel not in specs:
            kernel = "python3"
    except Exception:
        kernel = "python3"

    client = NotebookClient(
        run_nb,
        timeout=args.timeout,
        kernel_name=kernel,
        resources={"metadata": {"path": str(ROOT)}},
    )

    print(f"Executing cells 0..{args.upto} with .venv Python...", flush=True)
    # Force this venv's kernel by setting python executable via kernel — use python3 from path
    # Ensure sys.executable is the venv
    print("sys.executable =", sys.executable, flush=True)

    try:
        client.execute()
    except Exception as e:
        print("EXEC ERROR:", e, flush=True)
        # Still copy whatever outputs we got
        pass

    # Copy outputs/execution_count back into the real notebook for executed code cells
    for i in range(0, args.upto + 1):
        src_cell = run_nb.cells[i]
        dst_cell = nb.cells[i]
        if dst_cell.cell_type != "code":
            continue
        # If we temporarily changed pip cell source for exec, restore original from disk... 
        # Original already in dst_cell.source. Only copy outputs.
        dst_cell.outputs = src_cell.outputs
        dst_cell.execution_count = src_cell.execution_count

    nbformat.write(nb, NB_PATH)
    print(f"Saved outputs into {NB_PATH}", flush=True)

    # Print text/plain outputs for quick chat feedback
    for i in range(args.start, args.upto + 1):
        cell = nb.cells[i]
        if cell.cell_type != "code":
            continue
        texts = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                texts.append("".join(out.get("text", [])))
            elif out.get("output_type") == "execute_result":
                data = out.get("data", {})
                if "text/plain" in data:
                    texts.append("".join(data["text/plain"]) if isinstance(data["text/plain"], list) else data["text/plain"])
            elif out.get("output_type") == "error":
                texts.append("ERROR: " + "\n".join(out.get("traceback", [])[-3:]))
        if texts:
            print(f"\n----- cell {i} -----")
            print("".join(texts)[:2000])


if __name__ == "__main__":
    main()
