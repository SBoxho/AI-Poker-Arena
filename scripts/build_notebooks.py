"""Build the research notebooks from their py:percent sources, optionally executing them.

Notebooks are *authored* as plain Python files ``notebooks/src/<name>.py`` (diff-friendly, reviewable,
lintable) and *published* as ``notebooks/<name>.ipynb`` (what GitHub renders, with the executed outputs
embedded). Source format ("py:percent", the jupytext flavour):

    # %% [markdown]          <- markdown cell; every line is prefixed by "# " (a bare "#" is an empty line)
    # # Title
    # Some *text*.
    # %%                     <- code cell: everything up to the next marker
    x = 1

    .venv/Scripts/python scripts/build_notebooks.py                       # convert every source (no execution)
    .venv/Scripts/python scripts/build_notebooks.py --execute             # ... and execute them, outputs embedded
    .venv/Scripts/python scripts/build_notebooks.py --execute 03_genetic_algorithm_poker_agent
    .venv/Scripts/python scripts/build_notebooks.py --execute --timeout 1800 03_genetic_algorithm_poker_agent

Execution uses ``nbclient`` with a kernel that runs THIS interpreter (so run the script with the project's
``.venv/Scripts/python``), the repository root as working directory, and fails loudly - cell source plus
traceback - on the first error. A failed run never overwrites an existing ``.ipynb``.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import nbformat
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_raw_cell

# ==========================================
# BUILD CONFIGURATION
# ==========================================
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "notebooks" / "src"  # py:percent sources
OUTPUT_DIR = REPO_ROOT / "notebooks"  # generated .ipynb files
CELL_TIMEOUT_S = 900  # default per-cell execution timeout (seconds); override with --timeout
STARTUP_TIMEOUT_S = 120  # kernel start-up timeout (seconds)
MAX_NOTEBOOK_MB = 6.0  # refuse to write an .ipynb larger than this (embedded outputs must stay small)
KERNEL_NAME = "python3"
SOURCE_SUFFIX = ".py"

_MARKER = re.compile(r"^# %%(?P<rest>.*)$")
_KINDS = {"markdown": "markdown", "md": "markdown", "raw": "raw"}
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BYTES_PER_MB = 1024 * 1024


class NotebookBuildError(RuntimeError):
    """A notebook could not be converted or executed (the message says which cell and why)."""


# --------------------------------------------------------------------------------------
# py:percent <-> notebook
# --------------------------------------------------------------------------------------
def _cell_kind(rest: str) -> str:
    """``"markdown"``, ``"raw"`` or ``"code"`` from the text after ``# %%`` (``[markdown]`` / ``[md]`` / ``[raw]``)."""
    for tag in re.findall(r"\[(\w+)\]", rest):
        if tag in _KINDS:
            return _KINDS[tag]
    return "code"


def _markdown_body(lines: Sequence[str], first_line: int) -> str:
    """Strip the ``# `` comment prefix of a markdown cell; a line without ``#`` is a source error."""
    out = []
    for offset, line in enumerate(lines):
        if line.strip() == "":
            out.append("")
        elif line.startswith("# "):
            out.append(line[2:])
        elif line.rstrip() == "#":
            out.append("")
        else:
            raise NotebookBuildError(
                f"line {first_line + offset}: markdown cells must be '# '-prefixed comments, got {line!r}"
            )
    return "\n".join(out)


def parse_percent(text: str) -> NotebookNode:
    """Convert py:percent source text into an (unexecuted) notebook with deterministic cell ids."""
    cells: list[tuple[str, list[str], int]] = []  # (kind, lines, first line number)
    for number, line in enumerate(text.splitlines(), start=1):
        marker = _MARKER.match(line)
        if marker:
            cells.append((_cell_kind(marker["rest"]), [], number + 1))
        elif cells:
            cells[-1][1].append(line)
        elif line.strip():
            raise NotebookBuildError(f"line {number}: content before the first '# %%' cell marker: {line!r}")
    if not cells:
        raise NotebookBuildError("no '# %%' cell marker found")

    nb_cells = []
    for index, (kind, lines, first_line) in enumerate(cells):
        body = _markdown_body(lines, first_line) if kind != "code" else "\n".join(lines)
        body = body.strip("\n")
        if not body.strip():
            continue  # an empty cell carries no information
        cell = {"markdown": new_markdown_cell, "raw": new_raw_cell, "code": new_code_cell}[kind](body)
        cell["id"] = f"cell-{len(nb_cells):03d}"  # stable ids: re-building never produces a noisy diff
        nb_cells.append(cell)
    notebook = new_notebook(cells=nb_cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": KERNEL_NAME}
    notebook.metadata["language_info"] = {"name": "python"}
    nbformat.validate(notebook)
    return notebook


def to_percent(notebook: NotebookNode) -> str:
    """Inverse of :func:`parse_percent` (outputs are dropped): notebook -> py:percent text."""
    blocks = []
    for cell in notebook.cells:
        source = cell.source
        if cell.cell_type == "code":
            blocks.append(f"# %%\n{source}")
        else:
            tag = "[markdown]" if cell.cell_type == "markdown" else "[raw]"
            body = "\n".join(f"# {line}" if line else "#" for line in source.split("\n"))
            blocks.append(f"# %% {tag}\n{body}")
    return "\n\n".join(blocks) + "\n"


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------
def _kernel_manager():  # noqa: ANN202 - jupyter_client types are only needed here
    """A kernel manager whose kernel is *this* interpreter (the venv's), whatever kernelspecs are installed."""
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpec, KernelSpecManager

    class _CurrentPythonSpecs(KernelSpecManager):
        def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
            return KernelSpec(
                argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}", "--Application.log_level=ERROR"],
                display_name="Python 3 (project venv)",
                language="python",
                env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            )

    return KernelManager(kernel_name=KERNEL_NAME, kernel_spec_manager=_CurrentPythonSpecs())


def _failing_cell(notebook: NotebookNode) -> tuple[int, NotebookNode] | None:
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and any(o.output_type == "error" for o in cell.get("outputs", [])):
            return index, cell
    return None


def _describe_failure(name: str, notebook: NotebookNode, error: BaseException) -> str:
    header = f"{name}: execution failed ({type(error).__name__})"
    failing = _failing_cell(notebook)
    if failing is None:  # e.g. a timeout: no error output was written, report the error text itself
        return f"{header}\n{_ANSI.sub('', str(error))}"
    index, cell = failing
    errors = [o for o in cell.outputs if o.output_type == "error"]
    trace = "\n".join(errors[0].traceback)
    return f"{header} in cell {index}\n----- cell source -----\n{cell.source}\n----- traceback -----\n{_ANSI.sub('', trace)}"


def execute_notebook(
    notebook: NotebookNode, *, name: str = "notebook", timeout: int = CELL_TIMEOUT_S, cwd: Path = REPO_ROOT
) -> NotebookNode:
    """Run every cell in a fresh kernel (cwd = repo root); outputs are embedded into ``notebook``."""
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError

    client = NotebookClient(
        notebook,
        km=_kernel_manager(),
        timeout=timeout,
        startup_timeout=STARTUP_TIMEOUT_S,
        allow_errors=False,
        record_timing=False,  # per-cell timestamps would make every build a noisy diff
        resources={"metadata": {"path": str(cwd)}},
    )
    try:
        client.execute()
    except (CellExecutionError, CellTimeoutError) as error:
        raise NotebookBuildError(_describe_failure(name, notebook, error)) from error
    return notebook


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------
def discover(source_dir: Path) -> list[str]:
    return sorted(p.stem for p in source_dir.glob(f"*{SOURCE_SUFFIX}") if not p.stem.startswith("_"))


def build_notebook(
    name: str,
    *,
    execute: bool,
    source_dir: Path = SOURCE_DIR,
    output_dir: Path = OUTPUT_DIR,
    timeout: int = CELL_TIMEOUT_S,
    max_mb: float = MAX_NOTEBOOK_MB,
    cwd: Path = REPO_ROOT,
) -> Path:
    """Convert (and optionally execute) ``source_dir/<name>.py`` into ``output_dir/<name>.ipynb``."""
    source = source_dir / f"{name}{SOURCE_SUFFIX}"
    if not source.is_file():
        raise NotebookBuildError(f"{source} does not exist; available: {discover(source_dir) or 'none'}")
    try:
        notebook = parse_percent(source.read_text(encoding="utf-8"))
    except NotebookBuildError as error:
        raise NotebookBuildError(f"{source.name}: {error}") from error
    if execute:
        execute_notebook(notebook, name=name, timeout=timeout, cwd=cwd)
    nbformat.validate(notebook)
    text = nbformat.writes(notebook)
    size_mb = len(text.encode("utf-8")) / _BYTES_PER_MB
    if size_mb > max_mb:
        raise NotebookBuildError(
            f"{name}.ipynb would be {size_mb:.1f} MB (limit {max_mb:.1f} MB): save big figures to "
            f"artifacts/figures/ and keep embedded outputs small (or raise --max-mb)."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{name}.ipynb"
    target.write_text(text, encoding="utf-8")
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="notebook stems (e.g. 03_genetic_algorithm_poker_agent); default: all")
    parser.add_argument("--execute", action="store_true", help="execute the notebooks and embed their outputs")
    parser.add_argument("--timeout", type=int, default=CELL_TIMEOUT_S, help="per-cell timeout in seconds")
    parser.add_argument("--max-mb", type=float, default=MAX_NOTEBOOK_MB, help="refuse to write larger .ipynb files")
    parser.add_argument("--src-dir", type=Path, default=SOURCE_DIR, help="folder with the py:percent sources")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="folder for the generated .ipynb files")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    names = [n.removesuffix(SOURCE_SUFFIX) for n in args.names] or discover(args.src_dir)
    if not names:
        print(f"No notebook sources found in {args.src_dir}", file=sys.stderr)
        return 1
    for name in names:
        started = time.perf_counter()
        try:
            target = build_notebook(
                name, execute=args.execute, source_dir=args.src_dir, output_dir=args.out_dir,
                timeout=args.timeout, max_mb=args.max_mb,
            )
        except NotebookBuildError as error:
            print(f"\nBUILD FAILED: {error}", file=sys.stderr)
            return 1
        mode = "executed" if args.execute else "converted"
        size_kb = target.stat().st_size / 1024
        print(f"{mode} {name}: {target.relative_to(REPO_ROOT) if target.is_relative_to(REPO_ROOT) else target} "
              f"({size_kb:.0f} KB, {time.perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
