"""Shared notebook, Kaggle environment, and visual-explainer utilities for Kestrel.

Designed to be reused across all Kestrel notebooks (dataset audit, tokenizer training,
pretraining runs, and benchmark evaluation) with seamless local-first / GitHub-fallback
asset resolution.
"""

from __future__ import annotations

import base64
import html
import mimetypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

DEFAULT_REPO_URL = "https://github.com/MohammedKamalAlsyd/kestrel.git"
DEFAULT_BRANCH = "main"
DEFAULT_RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/MohammedKamalAlsyd/kestrel/{DEFAULT_BRANCH}"
)
DEFAULT_IMAGES_SUBDIR = Path("docs") / "images"


def is_kaggle() -> bool:
    """Return True when running inside a Kaggle notebook environment."""
    return Path("/kaggle/working").exists()


def get_work_dir() -> Path:
    """Return the writable working directory (`/kaggle/working` on Kaggle, else CWD)."""
    return Path("/kaggle/working") if is_kaggle() else Path.cwd()


def find_repo_root(extra_candidates: Sequence[str | Path] | None = None) -> Path | None:
    """Locate the Kestrel repository root containing `src/kestrel`."""
    candidates: list[Path] = []
    if extra_candidates:
        candidates.extend(Path(p) for p in extra_candidates)

    # Check relative to this installed/imported module first (src/kestrel/utils/notebook.py -> parents[3])
    module_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            module_root,
            Path.cwd(),
            Path.cwd().parent,
            get_work_dir() / "kestrel",
            Path("/kaggle/input/kestrel"),
        ]
    )

    seen: set[Path] = set()
    for root in candidates:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "src" / "kestrel").is_dir():
            return resolved
    return None


def ensure_kestrel_repo(
    repo_url: str = DEFAULT_REPO_URL,
    branch: str = DEFAULT_BRANCH,
    work_dir: str | Path | None = None,
    update_if_exists: bool = False,
) -> Path | None:
    """Ensure the Kestrel repository is available locally and `src/` is on `sys.path`.

    Checks local/Kaggle paths first; if missing, performs a shallow `git clone` into
    `<work_dir>/kestrel`. Returns the repository root or `None` if cloning fails.
    """
    repo_root = find_repo_root()
    if repo_root is None:
        base_dir = Path(work_dir) if work_dir is not None else get_work_dir()
        clone_dir = base_dir / "kestrel"
        try:
            if (clone_dir / ".git").exists():
                if update_if_exists:
                    subprocess.check_call(["git", "-C", str(clone_dir), "pull", "--ff-only"])
            else:
                subprocess.check_call(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "--branch",
                        branch,
                        repo_url,
                        str(clone_dir),
                    ]
                )
            if (clone_dir / "src" / "kestrel").is_dir():
                repo_root = clone_dir
        except Exception:
            return None

    src_dir = str((repo_root / "src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return repo_root


def configure_hf_cache(cache_dir: str | Path) -> Path:
    """Configure low-disk Hugging Face streaming cache variables inside `cache_dir`."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_path)
    os.environ["HF_DATASETS_CACHE"] = str(cache_path / "datasets")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return cache_path


def get_hf_token(secret_name: str = "HF_TOKEN") -> str | None:
    """Retrieve a Hugging Face token from environment variables or Kaggle Secrets."""
    token = os.environ.get(secret_name) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    if is_kaggle():
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

            token = UserSecretsClient().get_secret(secret_name)
            if token:
                os.environ[secret_name] = token
                return token
        except Exception:
            return None
    return None


def resolve_asset_source(
    filename: str,
    subdir: str | Path = DEFAULT_IMAGES_SUBDIR,
    repo_root: str | Path | None = None,
    raw_base_url: str = DEFAULT_RAW_BASE_URL,
) -> tuple[str, str]:
    """Resolve an image/animation asset locally first, falling back to GitHub raw URL.

    Returns:
        Tuple of ``(origin, location)`` where ``origin`` is ``"local"`` or ``"github"``,
        and ``location`` is either a local filesystem path or a raw GitHub HTTPS URL.
    """
    rel_subdir = Path(subdir)
    search_roots: list[Path] = []

    if repo_root is not None:
        search_roots.append(Path(repo_root))
    else:
        detected_root = find_repo_root()
        if detected_root is not None:
            search_roots.append(detected_root)
        search_roots.extend(
            [
                Path.cwd(),
                Path.cwd().parent,
                get_work_dir() / "kestrel",
                Path("/kaggle/input/kestrel"),
            ]
        )

    for root in search_roots:
        candidate = root / rel_subdir / filename
        if candidate.is_file():
            return ("local", str(candidate.resolve()))

    posix_subdir = rel_subdir.as_posix().strip("/")
    remote_url = f"{raw_base_url.rstrip('/')}/{posix_subdir}/{filename.lstrip('/')}"
    return ("github", remote_url)


def build_explainer_html(
    filename: str,
    caption: str = "",
    subdir: str | Path = DEFAULT_IMAGES_SUBDIR,
    repo_root: str | Path | None = None,
    raw_base_url: str = DEFAULT_RAW_BASE_URL,
    max_width: str = "100%",
) -> tuple[str, str, str]:
    """Build self-contained HTML for an image or GIF (local base64 or GitHub raw URL).

    Returns:
        ``(html_markup, origin, location)``
    """
    origin, location = resolve_asset_source(
        filename=filename,
        subdir=subdir,
        repo_root=repo_root,
        raw_base_url=raw_base_url,
    )

    if origin == "local":
        mime_type, _ = mimetypes.guess_type(location)
        mime_type = mime_type or "image/png"
        b64_data = base64.b64encode(Path(location).read_bytes()).decode("ascii")
        img_src = f"data:{mime_type};base64,{b64_data}"
    else:
        img_src = location

    safe_caption = html.escape(caption)
    alt_text = safe_caption or html.escape(filename)
    caption_block = (
        f'<div style="font-size:13px; color:#475569; margin-top:6px;"><em>{safe_caption}</em></div>'
        if caption
        else ""
    )
    markup = (
        f'<div style="text-align:center; padding:8px 0;">'
        f'<img src="{img_src}" alt="{alt_text}" '
        f'style="max-width:{max_width}; border-radius:8px; border:1px solid #cbd5e1;" />'
        f"{caption_block}"
        f"</div>"
    )
    return markup, origin, location


def show_explainer_visual(
    filename: str,
    caption: str = "",
    subdir: str | Path = DEFAULT_IMAGES_SUBDIR,
    repo_root: str | Path | None = None,
    raw_base_url: str = DEFAULT_RAW_BASE_URL,
    max_width: str = "100%",
) -> str:
    """Display an explanatory diagram or animated GIF inside a Jupyter/Kaggle notebook.

    Looks up ``<subdir>/<filename>`` locally first (e.g. from the local workspace or a
    cloned repo on Kaggle), and automatically falls back to fetching from GitHub raw
    content if the file is not present locally.

    Returns:
        The resolved local path or GitHub raw URL used for rendering.
    """
    from IPython.display import HTML, display

    markup, _origin, location = build_explainer_html(
        filename=filename,
        caption=caption,
        subdir=subdir,
        repo_root=repo_root,
        raw_base_url=raw_base_url,
        max_width=max_width,
    )
    display(HTML(markup))
    return location
