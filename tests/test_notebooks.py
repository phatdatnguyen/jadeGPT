"""Keep the teaching notebooks runnable without training during every test run.

End-to-end execution belongs in the notebook validation workflow. These checks
catch broken notebook files, stale backend calls, and accidentally committed
outputs before users start a lesson.
"""

import ast
import inspect
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import nbformat
import pytest

import jadegpt


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "train-gpt.ipynb",
    "sample-gpt.ipynb",
    "finetune-gpt.ipynb",
    "sample-gpt2.ipynb",
)


@pytest.fixture(params=NOTEBOOKS)
def notebook(request):
    path = ROOT / request.param
    return path, nbformat.read(path, as_version=4)


def test_notebook_schema_and_python_kernel(notebook):
    _, document = notebook
    nbformat.validate(document)
    assert document.metadata.kernelspec.name == "python3"
    assert document.metadata.kernelspec.language == "python"
    assert any(cell.cell_type == "code" for cell in document.cells)
    assert any(cell.cell_type == "markdown" for cell in document.cells)


def test_notebook_outputs_are_cleared(notebook):
    """Generated text, local paths, and stale metrics should not ship as results."""
    path, document = notebook
    for index, cell in enumerate(document.cells):
        if cell.cell_type == "code":
            assert cell.execution_count is None, f"{path.name}, cell {index}: execution count"
            assert cell.outputs == [], f"{path.name}, cell {index}: stored output"


def test_notebook_code_and_backend_calls_are_valid(notebook):
    """Syntax and real function signatures catch notebook/API drift cheaply."""
    path, document = notebook
    backend_calls = 0
    for index, cell in enumerate(document.cells):
        if cell.cell_type != "code":
            continue
        location = f"{path.name}, cell {index}"
        tree = ast.parse(cell.source, filename=location)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not re.match(r"^[A-Za-z]:[\\/]", node.value), (
                    f"{location}: use repository-relative paths instead of {node.value!r}"
                )
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "jadegpt"
            ):
                continue
            backend_calls += 1
            function = getattr(jadegpt, node.func.attr, None)
            assert callable(function), f"{location}: jadegpt.{node.func.attr} is unavailable"
            # Calls with unpacking need runtime values; inspect ordinary calls.
            if any(isinstance(arg, ast.Starred) for arg in node.args):
                continue
            if any(keyword.arg is None for keyword in node.keywords):
                continue
            try:
                inspect.signature(function).bind(
                    *[None for _ in node.args],
                    **{keyword.arg: None for keyword in node.keywords},
                )
            except TypeError as error:
                pytest.fail(f"{location}: jadegpt.{node.func.attr}: {error}")
    assert backend_calls, f"{path.name} should use the shared backend"


def test_notebook_local_lesson_links_exist(notebook):
    path, document = notebook
    for cell in document.cells:
        if cell.cell_type != "markdown":
            continue
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", cell.source):
            link = urlsplit(target)
            if link.scheme or link.netloc or not link.path:
                continue
            destination = path.parent / unquote(link.path)
            assert destination.is_file(), f"{path.name}: broken lesson link {target!r}"
