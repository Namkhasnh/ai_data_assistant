from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

from app.bootstrap import ensure_project_root, project_root


def test_home_page_imports_successfully() -> None:
    module = importlib.import_module("app.Home")

    assert module.PROJECT_TITLE == "Private Local AI Data Standardization and Enrichment Assistant"
    assert "Upload" in module.pipeline_overview()
    assert "08 Export" in module.navigation_hint()


def test_home_page_does_not_import_backend_services() -> None:
    source = Path("app/Home.py").read_text(encoding="utf-8")

    assert "from services." not in source
    assert "import services." not in source
    assert "pd." not in source
    assert "read_csv" not in source
    assert "ArtifactStore" not in source


def test_bootstrap_ensures_project_root_on_path(monkeypatch) -> None:
    root = project_root()
    original_path = [path for path in sys.path if path != str(root)]
    monkeypatch.setattr(sys, "path", original_path)

    returned_root = ensure_project_root()

    assert returned_root == root
    assert sys.path[0] == str(root)


def test_streamlit_pages_import_with_bootstrap() -> None:
    for page_path in _streamlit_page_files():
        namespace = runpy.run_path(str(page_path))
        assert "main" in namespace


def test_pages_use_bootstrap_without_local_path_logic() -> None:
    for page_path in _streamlit_page_files():
        source = page_path.read_text(encoding="utf-8")
        assert "ensure_project_root()" in source
        assert "sys.path" not in source


def _streamlit_page_files() -> list[Path]:
    return sorted(path for path in Path("app/pages").glob("*.py") if path.name[0].isdigit())


def test_home_page_renders_without_artifacts(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    fake_streamlit = SimpleNamespace(
        set_page_config=lambda **kwargs: calls.append(("set_page_config", kwargs)),
        title=lambda value: calls.append(("title", value)),
        markdown=lambda value: calls.append(("markdown", value)),
        info=lambda value: calls.append(("info", value)),
        columns=lambda count: [FakeColumn(calls) for _ in range(count)],
        expander=lambda label: FakeExpander(calls, label),
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    module = importlib.import_module("app.Home")

    module.main()

    assert ("title", module.PROJECT_TITLE) in calls
    assert any(call[0] == "set_page_config" for call in calls)
    assert any(call == ("metric", ("Backend Status", "Stable")) for call in calls)


class FakeColumn:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls

    def metric(self, label: str, value: str) -> None:
        self.calls.append(("metric", (label, value)))


class FakeExpander:
    def __init__(self, calls: list[tuple[str, object]], label: str) -> None:
        self.calls = calls
        self.label = label

    def __enter__(self) -> FakeExpander:
        self.calls.append(("expander", self.label))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None
