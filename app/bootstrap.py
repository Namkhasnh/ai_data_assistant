from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """Return the repository root for the Streamlit application."""

    return Path(__file__).resolve().parents[1]


def ensure_project_root() -> Path:
    """Ensure imports like app.components work when Streamlit runs app/Home.py."""

    root = project_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root
