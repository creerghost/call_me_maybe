import pytest
import json
from pathlib import Path
from .models import FunctionCallResult


class OutputWriter():
    @staticmethod
    def write_output(results: list[FunctionCallResult],
                     output_path: str) -> None:
        if not output_path.strip():
            raise ValueError("Output path is not defined")

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        tmp = [item.model_dump() for item in results] if results else []

        with output_path_obj.open("w", encoding="utf-8") as file_handle:
            json.dump(tmp, file_handle, indent=2, ensure_ascii=False)
        print(f"[SAVE] Results have been saved to {output_path}.\n")


# === TESTS ===

def test_save_json_success(tmp_path: Path) -> None:

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
    fake_answers = [
        FunctionCallResult(
            prompt="test",
            name="test",
            parameters={}
        )
    ]
    with pytest.raises(ValueError, match="Output path is not defined"):
        OutputWriter.write_output(fake_answers, "   ")


if __name__ == "__main__":
    from pathlib import Path

    path_obj = Path(__file__)
    print(f"\n=== TESTING {path_obj.stem}.py ===\n")

    pytest.main(["-v", "-o", "python_classes=*Suite", __file__])
