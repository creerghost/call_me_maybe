"""Module for test_output.py."""

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
