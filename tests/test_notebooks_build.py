"""Notebook infrastructure: py:percent conversion, execution (venv kernel, repo-root cwd, loud failures)
and the shared plotting helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import matplotlib
import nbformat
import pytest

from poker_arena.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "build_notebooks.py"
NOTEBOOK_SOURCES = REPO_ROOT / "notebooks" / "src"

SYNTHETIC_SOURCE = """\
# %% [markdown]
# # Title
#
# Reproduce: `.venv/Scripts/python scripts/build_notebooks.py`
# - item one
# %%
import os
import sys

print(os.getcwd())
print(sys.executable)
value = 6 * 7
value
# %% [markdown]
# Second *markdown* cell
# %%
print("done")
"""


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_notebooks_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass-free module, but keeps pickling/introspection happy
    spec.loader.exec_module(module)
    return module


def _write(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(text, encoding="utf-8")
    return path


def _stream_text(notebook: nbformat.NotebookNode, cell_index: int) -> str:
    return "".join(o.text for o in notebook.cells[cell_index].outputs if o.output_type == "stream")


# ---------------------------------------------------------------------------- conversion
def test_parse_percent_cells_and_markdown_prefix(builder: ModuleType) -> None:
    notebook = builder.parse_percent(SYNTHETIC_SOURCE)
    kinds = [c.cell_type for c in notebook.cells]
    assert kinds == ["markdown", "code", "markdown", "code"]
    assert notebook.cells[0].source == "# Title\n\nReproduce: `.venv/Scripts/python scripts/build_notebooks.py`\n- item one"
    assert notebook.cells[1].source.splitlines()[0] == "import os"
    assert notebook.cells[1].source.endswith("value")  # no trailing blank line
    assert [c.id for c in notebook.cells] == ["cell-000", "cell-001", "cell-002", "cell-003"]  # deterministic
    nbformat.validate(notebook)


def test_round_trip_is_lossless(builder: ModuleType) -> None:
    notebook = builder.parse_percent(SYNTHETIC_SOURCE)
    text = builder.to_percent(notebook)
    assert builder.parse_percent(text).cells == notebook.cells
    assert builder.to_percent(builder.parse_percent(text)) == text  # idempotent normal form


def test_build_without_execution_writes_a_valid_notebook(builder: ModuleType, tmp_path: Path) -> None:
    _write(tmp_path / "src", "tiny", SYNTHETIC_SOURCE)
    target = builder.build_notebook("tiny", execute=False, source_dir=tmp_path / "src", output_dir=tmp_path / "out")
    notebook = nbformat.read(target, as_version=4)
    nbformat.validate(notebook)
    assert len(notebook.cells) == 4
    assert all(not c.get("outputs") for c in notebook.cells if c.cell_type == "code")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("print(1)\n# %%\nx = 1\n", "before the first"),
        ("# %% [markdown]\nnot a comment\n", "'# '-prefixed"),
        ("just text\n", "before the first"),
        ("", "no '# %%' cell marker"),
    ],
)
def test_invalid_sources_are_rejected_loudly(builder: ModuleType, source: str, message: str) -> None:
    with pytest.raises(builder.NotebookBuildError, match=message):
        builder.parse_percent(source)


def test_unknown_name_lists_the_available_notebooks(builder: ModuleType, tmp_path: Path) -> None:
    _write(tmp_path, "present", "# %%\nx = 1\n")
    with pytest.raises(builder.NotebookBuildError, match="present"):
        builder.build_notebook("absent", execute=False, source_dir=tmp_path, output_dir=tmp_path)


def test_discover_ignores_private_files(builder: ModuleType, tmp_path: Path) -> None:
    for name in ("b_nb", "a_nb", "_scratch"):
        _write(tmp_path, name, "# %%\nx = 1\n")
    assert builder.discover(tmp_path) == ["a_nb", "b_nb"]


def test_oversized_notebooks_are_refused_and_nothing_is_written(builder: ModuleType, tmp_path: Path) -> None:
    _write(tmp_path / "src", "big", "# %%\nx = 1\n")
    with pytest.raises(builder.NotebookBuildError, match="limit"):
        builder.build_notebook("big", execute=False, source_dir=tmp_path / "src", output_dir=tmp_path / "out", max_mb=1e-6)
    assert not (tmp_path / "out" / "big.ipynb").exists()


def test_every_committed_notebook_source_parses(builder: ModuleType) -> None:
    sources = sorted(NOTEBOOK_SOURCES.glob("*.py")) if NOTEBOOK_SOURCES.exists() else []
    for source in sources:
        notebook = builder.parse_percent(source.read_text(encoding="utf-8"))
        assert any(c.cell_type == "code" for c in notebook.cells), source.name
        first_markdown = next(c for c in notebook.cells if c.cell_type == "markdown")
        assert "Reproduce" in first_markdown.source, f"{source.name}: first markdown cell needs a 'Reproduce:' line"
        first_code = next(c for c in notebook.cells if c.cell_type == "code")
        assert "NOTEBOOK CONFIGURATION" in first_code.source, f"{source.name}: first code cell must be the config block"


# ---------------------------------------------------------------------------- execution
def test_execution_uses_the_venv_kernel_repo_root_and_embeds_outputs(builder: ModuleType, tmp_path: Path) -> None:
    _write(tmp_path / "src", "tiny", SYNTHETIC_SOURCE)
    target = builder.build_notebook("tiny", execute=True, source_dir=tmp_path / "src", output_dir=tmp_path / "out", timeout=120)
    notebook = nbformat.read(target, as_version=4)
    printed = _stream_text(notebook, 1).splitlines()
    assert Path(printed[0]).resolve() == REPO_ROOT.resolve()  # working directory = repo root
    assert Path(printed[1]).resolve() == Path(sys.executable).resolve()  # this venv's interpreter
    result = [o for o in notebook.cells[1].outputs if o.output_type == "execute_result"]
    assert result and result[0].data["text/plain"] == "42"
    assert _stream_text(notebook, 3).strip() == "done"
    assert "execution" not in notebook.cells[1].metadata  # no timestamps: rebuilding gives stable diffs


def test_failing_cell_fails_loudly_with_source_and_traceback_and_keeps_old_output(builder: ModuleType, tmp_path: Path) -> None:
    _write(tmp_path / "src", "broken", "# %%\nok = 1\n# %%\nresult = 1 / 0  # the failing line\n# %%\nunreached = 2\n")
    out = tmp_path / "out"
    out.mkdir()
    previous = out / "broken.ipynb"
    previous.write_text("previous good build", encoding="utf-8")
    with pytest.raises(builder.NotebookBuildError) as caught:
        builder.build_notebook("broken", execute=True, source_dir=tmp_path / "src", output_dir=out, timeout=120)
    message = str(caught.value)
    assert "cell 1" in message and "result = 1 / 0" in message  # which cell, and its source
    assert "ZeroDivisionError" in message and "Traceback" in message
    assert "\x1b[" not in message  # ANSI colour codes stripped
    assert previous.read_text(encoding="utf-8") == "previous good build"


def test_cli_returns_nonzero_on_failure_and_zero_on_success(builder: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path / "src", "good", "# %%\nx = 1\n")
    _write(tmp_path / "src", "bad", "# %% [markdown]\nbroken markdown\n")
    args = ["--src-dir", str(tmp_path / "src"), "--out-dir", str(tmp_path / "out")]
    assert builder.main([*args, "good"]) == 0
    assert (tmp_path / "out" / "good.ipynb").exists()
    assert builder.main([*args, "bad"]) == 1
    assert "BUILD FAILED" in capsys.readouterr().err


# ---------------------------------------------------------------------------- plotting helpers
@pytest.fixture
def plots(tmp_path: Path):
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from poker_arena.explain import notebook_plots

    notebook_plots.apply_style()
    yield notebook_plots
    plt.close("all")


def test_plot_helpers_render_and_save_pngs(plots, tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["random", "teacher", "nit"]
    estimates, low, high = [150.0, -56.0, 1.0], [80.0, -100.0, -26.0], [230.0, -12.0, 28.0]
    fig, (ax_forest, ax_bar) = plt.subplots(1, 2)
    plots.forest_plot(ax_forest, labels, {"evolved": (estimates, low, high), "tag": (high, estimates, high)}, x_label="BB/100")
    plots.bar_with_ci(ax_bar, labels, estimates, low, high)
    plots.title(ax_bar, "Title", "subtitle")
    plots.footnote(fig, "note")
    path = plots.savefig("unit_test_figure", fig, directory=tmp_path)
    assert path == tmp_path / "unit_test_figure.png"
    assert path.stat().st_size > 1000
    assert plt.fignum_exists(fig.number)  # still open, so the notebook displays it inline


@pytest.mark.parametrize("bad_name", ["../escape", "with.png", "sub/dir"])
def test_savefig_rejects_paths_and_extensions(plots, bad_name: str, tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    with pytest.raises(ValueError, match="bare file stem"):
        plots.savefig(bad_name, directory=tmp_path)


def test_palette_is_the_documented_fixed_order(plots) -> None:
    assert plots.ACCENT == plots.SERIES[0]
    assert len(set(plots.SERIES)) == len(plots.SERIES)
