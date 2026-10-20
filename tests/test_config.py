"""Tests for cardiomics_s2f.config (path loading and API key lookup)."""

import os
from pathlib import Path

import pytest

from cardiomics_s2f import config


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Runs every test in an empty folder without user config leaking in."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(config.PATHS_ENV_VAR, raising=False)
    monkeypatch.delenv(config.API_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(config, "_REPO_ROOT", tmp_path / "no_repo")
    monkeypatch.setattr(config, "_find_dotenv", lambda: None)


def write_paths(folder: Path, text: str) -> Path:
    path = folder / "config" / "paths.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- load_paths ---------------------------------------------------------------

def test_relative_entries_resolve_against_data_root(tmp_path):
    source = write_paths(tmp_path, "data_root: ..\ngenome_fasta: ref/genome.fa\n")
    paths = config.load_paths(source)
    assert paths["data_root"] == Path(os.path.normpath(tmp_path))
    assert paths.genome_fasta == Path(os.path.normpath(tmp_path / "ref" / "genome.fa"))
    assert paths.source == source


def test_absolute_entries_are_kept(tmp_path):
    absolute = tmp_path / "elsewhere" / "ckpt"
    source = write_paths(tmp_path, f"data_root: ..\nckpt: {absolute.as_posix()}\n")
    assert config.load_paths(source)["ckpt"] == Path(os.path.normpath(absolute))


def test_absolute_data_root(tmp_path):
    root = tmp_path / "data"
    source = write_paths(tmp_path, f"data_root: {root.as_posix()}\nbams: raw_bam\n")
    assert config.load_paths(source)["bams"] == Path(os.path.normpath(root / "raw_bam"))


def test_data_root_defaults_to_config_folder(tmp_path):
    source = write_paths(tmp_path, "gtf: genes.gtf\n")
    paths = config.load_paths(source)
    assert paths["data_root"] == Path(os.path.normpath(tmp_path / "config"))
    assert paths["gtf"] == Path(os.path.normpath(tmp_path / "config" / "genes.gtf"))


def test_environment_variables_are_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("CS2F_TEST_SCRATCH", str(tmp_path / "scratch"))
    source = write_paths(tmp_path, "data_root: ..\nnode_tmp_dir: $CS2F_TEST_SCRATCH/tmp\n")
    assert config.load_paths(source)["node_tmp_dir"] == Path(os.path.normpath(tmp_path / "scratch" / "tmp"))


def test_unknown_key_lists_available_keys(tmp_path):
    paths = config.load_paths(write_paths(tmp_path, "data_root: ..\ngtf: genes.gtf\n"))
    with pytest.raises(KeyError, match="Available keys: data_root, gtf"):
        paths["missing"]
    with pytest.raises(AttributeError, match="missing"):
        paths.missing


def test_non_string_entry_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="'batch_size'"):
        config.load_paths(write_paths(tmp_path, "data_root: ..\nbatch_size: 16\n"))


def test_non_mapping_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="mapping"):
        config.load_paths(write_paths(tmp_path, "- a\n- b\n"))


# --- find_paths_file ----------------------------------------------------------

def test_env_var_is_used(tmp_path, monkeypatch):
    source = write_paths(tmp_path / "custom", "data_root: ..\n")
    monkeypatch.setenv(config.PATHS_ENV_VAR, str(source))
    assert config.find_paths_file() == source


def test_explicit_path_beats_env_var(tmp_path, monkeypatch):
    explicit = write_paths(tmp_path / "a", "data_root: ..\n")
    monkeypatch.setenv(config.PATHS_ENV_VAR, str(write_paths(tmp_path / "b", "data_root: ..\n")))
    assert config.find_paths_file(explicit) == explicit


def test_search_upwards_from_cwd(tmp_path, monkeypatch):
    source = write_paths(tmp_path, "data_root: ..\n")
    nested = tmp_path / "analysis_notebooks" / "vep"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert config.find_paths_file() == source


def test_falls_back_to_repo_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    source = write_paths(repo, "data_root: ..\n")
    monkeypatch.setattr(config, "_REPO_ROOT", repo)
    assert config.find_paths_file() == source


def test_missing_explicit_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        config.find_paths_file(tmp_path / "nope.yaml")


def test_no_file_found_explains_setup():
    with pytest.raises(FileNotFoundError, match="paths.example.yaml"):
        config.find_paths_file()


def test_example_template_loads():
    template = Path(__file__).resolve().parents[1] / "config" / "paths.example.yaml"
    paths = config.load_paths(template)
    assert paths["data_root"] == Path(os.path.normpath(template.parents[1]))
    assert paths["genome_fasta"].name == "genome.fa"


# --- get_alphagenome_api_key --------------------------------------------------

def test_explicit_key_wins(monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV_VAR, "from-env")
    assert config.get_alphagenome_api_key("explicit") == "explicit"


def test_key_from_environment(monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV_VAR, "  from-env  ")
    assert config.get_alphagenome_api_key() == "from-env"


def test_key_from_dotenv_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{config.API_KEY_ENV_VAR}=from-file\n", encoding="utf-8")
    assert config.get_alphagenome_api_key(env_file=env_file) == "from-file"
    assert config.API_KEY_ENV_VAR not in os.environ


def test_key_from_discovered_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{config.API_KEY_ENV_VAR}=discovered\n", encoding="utf-8")
    monkeypatch.setattr(config, "_find_dotenv", lambda: env_file)
    assert config.get_alphagenome_api_key() == "discovered"


def test_environment_beats_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{config.API_KEY_ENV_VAR}=from-file\n", encoding="utf-8")
    monkeypatch.setenv(config.API_KEY_ENV_VAR, "from-env")
    assert config.get_alphagenome_api_key(env_file=env_file) == "from-env"


def test_empty_values_are_ignored(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{config.API_KEY_ENV_VAR}=\n", encoding="utf-8")
    monkeypatch.setenv(config.API_KEY_ENV_VAR, "   ")
    with pytest.raises(RuntimeError, match="No AlphaGenome API key"):
        config.get_alphagenome_api_key("", env_file=env_file)


def test_missing_key_explains_setup():
    with pytest.raises(RuntimeError, match=r"\.env\.example"):
        config.get_alphagenome_api_key()
