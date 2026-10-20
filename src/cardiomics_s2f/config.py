"""
Project Configuration
=====================

Central access to machine-specific data paths and API credentials, so that no
library code or workflow script hard-codes either.

* **Paths** are read from a YAML file (``config/paths.yaml``; the committed
  template is ``config/paths.example.yaml``). Relative ``data_root`` resolves
  against the folder of the YAML file, every other relative entry against
  ``data_root``. ``~`` and environment variables (e.g. ``$SCRATCH``) are expanded.
* **Credentials** are read from the environment or a local, gitignored ``.env``
  file (template: ``.env.example``). They are never passed on the command line,
  where they would show up in ``ps`` output and SLURM logs.
"""

import os
from pathlib import Path
from typing import Iterator, Mapping, Optional, Union

import yaml

PATHS_ENV_VAR = "CS2F_PATHS"
API_KEY_ENV_VAR = "ALPHAGENOME_API_KEY"
DEFAULT_PATHS_FILE = Path("config") / "paths.yaml"

# Repository root for editable installs (src/cardiomics_s2f/config.py -> repo/)
_REPO_ROOT = Path(__file__).resolve().parents[2]

PathLike = Union[str, os.PathLike]


class ProjectPaths(Mapping):
    """Resolved project paths with dictionary and attribute access.

    Both ``paths["genome_fasta"]`` and ``paths.genome_fasta`` return a ``Path``.
    Unknown keys raise an error that names the config file and lists the
    available keys.

    Attributes:
        source (Path): The YAML file the paths were loaded from.
    """

    def __init__(self, paths: dict, source: Path):
        """Initializes the mapping from already resolved paths."""
        self._paths = paths
        self.source = source

    def __getitem__(self, key: str) -> Path:
        try:
            return self._paths[key]
        except KeyError:
            raise KeyError(
                f"'{key}' is not defined in {self.source}. "
                f"Available keys: {', '.join(sorted(self._paths))}"
            ) from None

    def __getattr__(self, key: str) -> Path:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self[key]
        except KeyError as err:
            raise AttributeError(err.args[0]) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __repr__(self) -> str:
        return f"ProjectPaths(source={str(self.source)!r}, keys={sorted(self._paths)})"


def _expand(value: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(value.strip())))


def _normalise(path: Path) -> Path:
    # Collapses ".." without following symlinks (data folders are often symlinked)
    return Path(os.path.normpath(path))


def find_paths_file(path: Optional[PathLike] = None) -> Path:
    """Locates the paths YAML file.

    The first match wins:

    1. ``path``, if given.
    2. The file named by the ``CS2F_PATHS`` environment variable.
    3. ``config/paths.yaml`` in the current working directory or any parent.
    4. ``config/paths.yaml`` in the repository of an editable install.

    Args:
        path (str or os.PathLike, optional): Explicit path to a paths YAML file.

    Returns:
        Path: The paths YAML file.

    Raises:
        FileNotFoundError: If an explicitly requested file does not exist, or
            no file is found at all.
    """
    explicit = path if path is not None else os.environ.get(PATHS_ENV_VAR)
    if explicit:
        explicit = _expand(str(explicit))
        if not explicit.is_file():
            raise FileNotFoundError(f"Paths file not found: {explicit}")
        return explicit

    cwd = Path.cwd()
    for folder in (cwd, *cwd.parents, _REPO_ROOT):
        candidate = folder / DEFAULT_PATHS_FILE
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No {DEFAULT_PATHS_FILE.as_posix()} found. Copy config/paths.example.yaml to "
        f"config/paths.yaml and adapt it, or point ${PATHS_ENV_VAR} to a paths file."
    )


def load_paths(path: Optional[PathLike] = None) -> ProjectPaths:
    """Loads and resolves the project paths.

    Args:
        path (str or os.PathLike, optional): Explicit path to a paths YAML file.
            Defaults to the lookup order of `find_paths_file`.

    Returns:
        ProjectPaths: Mapping of path names to resolved ``Path`` objects. It
        always contains ``data_root``.

    Raises:
        FileNotFoundError: If no paths file can be found.
        ValueError: If the file is not a mapping or an entry is not a string.
    """
    source = find_paths_file(path)
    with open(source, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must contain a mapping of names to paths.")

    for key, value in raw.items():
        if not isinstance(value, str):
            raise ValueError(f"'{key}' in {source} must be a path string, got {type(value).__name__}.")

    data_root = _expand(raw.get("data_root", "."))
    if not data_root.is_absolute():
        data_root = source.parent / data_root
    resolved = {"data_root": _normalise(data_root)}

    for key, value in raw.items():
        if key == "data_root":
            continue
        entry = _expand(value)
        resolved[key] = _normalise(entry if entry.is_absolute() else data_root / entry)

    return ProjectPaths(resolved, source)


def _find_dotenv() -> Optional[Path]:
    from dotenv import find_dotenv

    found = find_dotenv(usecwd=True)
    return Path(found) if found else None


def get_alphagenome_api_key(api_key: Optional[str] = None, env_file: Optional[PathLike] = None) -> str:
    """Returns the AlphaGenome API key.

    The first non-empty value wins:

    1. ``api_key``, if given.
    2. The ``ALPHAGENOME_API_KEY`` environment variable.
    3. ``ALPHAGENOME_API_KEY`` in ``env_file``, or in the nearest ``.env`` file
       found from the current working directory upwards.

    The ``.env`` file is only read; ``os.environ`` is not modified.

    Args:
        api_key (str, optional): Key passed explicitly by the caller.
        env_file (str or os.PathLike, optional): Explicit ``.env`` file to read.

    Returns:
        str: The API key.

    Raises:
        RuntimeError: If no key is found.
    """
    if api_key and api_key.strip():
        return api_key.strip()

    from_env = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if from_env:
        return from_env

    dotenv_path = _expand(str(env_file)) if env_file is not None else _find_dotenv()
    if dotenv_path is not None and dotenv_path.is_file():
        from dotenv import dotenv_values

        from_file = (dotenv_values(dotenv_path).get(API_KEY_ENV_VAR) or "").strip()
        if from_file:
            return from_file

    raise RuntimeError(
        f"No AlphaGenome API key found. Set the {API_KEY_ENV_VAR} environment variable "
        f"or add '{API_KEY_ENV_VAR}=<key>' to a .env file (see .env.example)."
    )
