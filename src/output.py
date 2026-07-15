import json
from pathlib import Path
from .models import FunctionCallResult


class OutputWriter:
    """Handles writing the final structured JSON results."""

    @staticmethod
    def write_output(
        results: list[FunctionCallResult], output_path: str
    ) -> None:
        """Saves a list of FunctionCallResults to a JSON file.

        Args:
            results (list[FunctionCallResult]): The parsed function call data.
            output_path (str): The file path where the JSON should be written.

        Raises:
            ValueError: If the output path is empty or consists of whitespace.
        """
        if not output_path.strip():
            raise ValueError("Output path is not defined")

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        tmp = [item.model_dump() for item in results] if results else []

        with output_path_obj.open("w", encoding="utf-8") as file_handle:
            json.dump(tmp, file_handle, indent=2, ensure_ascii=False)
