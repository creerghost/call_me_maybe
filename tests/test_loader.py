"""Module for test_loader.py."""

import pytest
import json
import os
from pathlib import Path
from src.catch import LoaderError
from src.loader import Loader


def test_loader_valid_files_existing() -> None:
    """Tests loading of valid JSON configuration files from disk."""
    fdef = "data/input/functions_definition.json"
    fcall = "data/input/function_calling_tests.json"

    if not os.path.exists(fdef) or not os.path.exists(fcall):
        pytest.skip("Default files not found, skipping hardcoded test.")

    loader = Loader(fdef_name=fdef, fcall_name=fcall)
    assert len(loader.fn_defs) > 0
    assert len(loader.test_prompts) > 0


def test_loader_valid_files_created(tmp_path: Path) -> None:
    """Tests loading dynamically created valid JSON configuration files.

    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
    """
    fdef: Path = tmp_path / "valid_def.json"
    valid_def_data = [{
        "name": "fn_test",
        "description": "test function",
        "parameters": {"a": {"type": "string"}},
        "returns": {"type": "string"}
    }]
    fdef.write_text(json.dumps(valid_def_data))

    fcall: Path = tmp_path / "valid_call.json"
    valid_call_data = [{"prompt": "test prompt"}]
    fcall.write_text(json.dumps(valid_call_data))

    loader = Loader(str(fdef), str(fcall))

    assert len(loader.fn_defs) == 1
    assert len(loader.test_prompts) == 1
    assert loader.fn_defs[0].name == "fn_test"
    assert loader.test_prompts[0].prompt == "test prompt"


def test_loader_file_not_found() -> None:
    """Tests that a LoaderError is raised when a file does not exist."""
    with pytest.raises(LoaderError):
        Loader("does_not_exist.json", "data/input/function_calling_tests.json")


# tmp path is built-in pytest variable
def test_loader_broken_json(tmp_path: Path) -> None:
    """Tests that a LoaderError is raised when JSON syntax is invalid.

    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
    """
    broken_file: Path = tmp_path / "broken.json"
    broken_file.write_text("{ this is not a valid json object }")

    with pytest.raises(LoaderError):
        Loader(str(broken_file), str(broken_file))


def test_loader_broken_json_empty(tmp_path: Path) -> None:
    """Tests that a LoaderError is raised when JSON objects are empty.

    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
    """
    broken_file: Path = tmp_path / "broken.json"
    broken_file.write_text("{}")

    with pytest.raises(LoaderError):
        Loader(str(broken_file), str(broken_file))
