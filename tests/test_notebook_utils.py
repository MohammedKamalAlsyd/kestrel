"""Unit tests for shared notebook utilities in kestrel.utils.notebook."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from kestrel.utils import notebook as nb_utils


def test_find_repo_root_detects_workspace():
    root = nb_utils.find_repo_root()
    assert root is not None
    assert (root / "src" / "kestrel" / "utils" / "notebook.py").is_file()


def test_resolve_asset_source_prefers_local_file(tmp_path: Path):
    img_dir = tmp_path / "docs" / "images"
    img_dir.mkdir(parents=True)
    local_img = img_dir / "diagram.png"
    local_img.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    origin, loc = nb_utils.resolve_asset_source(
        "diagram.png",
        repo_root=tmp_path,
    )
    assert origin == "local"
    assert Path(loc) == local_img.resolve()


def test_resolve_asset_source_falls_back_to_github_raw(tmp_path: Path):
    origin, loc = nb_utils.resolve_asset_source(
        "missing_chart.gif",
        repo_root=tmp_path,
    )
    assert origin == "github"
    assert loc == (
        "https://raw.githubusercontent.com/MohammedKamalAlsyd/kestrel/main/"
        "docs/images/missing_chart.gif"
    )


def test_build_explainer_html_local_and_github(tmp_path: Path):
    # Remote GitHub fallback HTML
    remote_html, origin, loc = nb_utils.build_explainer_html(
        "01_kestrel_data_pipeline.png",
        caption="Pipeline Overview",
        repo_root=tmp_path,
    )
    assert origin == "github"
    assert loc.startswith("https://raw.githubusercontent.com/MohammedKamalAlsyd/kestrel/main/")
    assert f'src="{loc}"' in remote_html
    assert "Pipeline Overview" in remote_html

    # Local base64 HTML (PNG & GIF)
    img_dir = tmp_path / "docs" / "images"
    img_dir.mkdir(parents=True)
    gif_file = img_dir / "demo.gif"
    gif_file.write_bytes(b"GIF89a_demo")

    local_html, local_origin, local_loc = nb_utils.build_explainer_html(
        "demo.gif",
        caption="Animated Demo",
        repo_root=tmp_path,
    )
    assert local_origin == "local"
    assert Path(local_loc) == gif_file.resolve()
    assert "data:image/gif;base64," in local_html
    assert "Animated Demo" in local_html


def test_show_explainer_visual_calls_ipython_display(tmp_path: Path):
    with patch("IPython.display.display") as mock_display:
        loc = nb_utils.show_explainer_visual(
            "01_kestrel_data_pipeline.png",
            caption="Test",
            repo_root=tmp_path,
        )
    assert loc.endswith("/docs/images/01_kestrel_data_pipeline.png")
    mock_display.assert_called_once()
