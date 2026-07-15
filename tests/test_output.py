import pytest
import json
from pathlib import Path
from src.models import FunctionCallResult
from src.output import OutputWriter


def test_save_json_success(tmp_path: Path) -> None:
    """Tests that save json success."""
    out_file: Path = tmp_path / "answers.json"

    fake_answers = [
        FunctionCallResult(
            prompt="What is the weather?",
            name="get_weather",
            parameters={"location": "Paris"}
        )
    ]

    OutputWriter.write_output(fake_answers, str(out_file))
    assert out_file.exists()
    saved_data = json.loads(out_file.read_text())

    assert len(saved_data) == 1
    assert saved_data[0]["name"] == "get_weather"


def test_save_json_empty_path() -> None:
    """Tests that save json empty path."""
    fake_answers = [
        FunctionCallResult(
            prompt="test",
            name="test",
            parameters={}
        )
    ]
    with pytest.raises(ValueError, match="Output path is not defined"):
        OutputWriter.write_output(fake_answers, "   ")


def test_save_json_empty_results(tmp_path: Path) -> None:
    """Tests that an empty results list writes an empty JSON array.

    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
    """
    out_file: Path = tmp_path / "empty.json"
    OutputWriter.write_output([], str(out_file))
    data = json.loads(out_file.read_text())
    assert data == []


def test_save_json_nested_directories(tmp_path: Path) -> None:
    """Tests that intermediate directories are created for output.

    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
    """
    out_file: Path = tmp_path / "a" / "b" / "c" / "results.json"
    results = [
        FunctionCallResult(
            prompt="test", name="fn_test", parameters={}
        )
    ]
    OutputWriter.write_output(results, str(out_file))
    assert out_file.exists()


def test_save_json_unicode_content(tmp_path: Path) -> None:
    """Tests that unicode content is written correctly to JSON output.

    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
    """
    out_file: Path = tmp_path / "unicode.json"
    results = [
        FunctionCallResult(
            prompt="Привет мир",
            name="fn_greet",
            parameters={"name": "Алексей"},
        )
    ]
    OutputWriter.write_output(results, str(out_file))
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data[0]["parameters"]["name"] == "Алексей"
