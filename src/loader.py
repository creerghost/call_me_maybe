import pytest
import json
from pydantic import ValidationError
from .models import FunctionDefinition, TestPrompt
from .catch import LoaderError


class Loader():
    def __init__(self, fdef_name: str, fcall_name: str) -> None:
        self.fn_defs: list[FunctionDefinition] = []
        self.test_prompts: list[TestPrompt] = []
        self._load(fdef_name, fcall_name)

    def _load(self, fdef: str, fcall: str) -> None:
        try:
            with open(fdef, "r") as f:
                json_defs = json.load(f)
            self.fn_defs = [FunctionDefinition(**d) for d in json_defs]
        except FileNotFoundError:
            raise LoaderError(f"File not found: {fdef}")
        except json.JSONDecodeError as e:
            raise LoaderError(f"Invalid JSON in {fdef}: {e}")
        except ValidationError as e:
            raise LoaderError(f"Schema mismatch in {fdef}:\n{e}")

        try:
            with open(fcall, "r") as f:
                json_prompts = json.load(f)
            self.test_prompts = [TestPrompt(**p) for p in json_prompts]
        except FileNotFoundError:
            raise LoaderError(f"File not found: {fcall}")
        except json.JSONDecodeError as e:
            raise LoaderError(f"Invalid JSON in {fcall}: {e}")
        except ValidationError as e:
            raise LoaderError(f"Schema mismatch in {fcall}:\n{e}")

# === TESTS ===

def test_loader_valid_files_existing() -> None:
    loader = Loader(
        fdef_name="data/input/functions_definition.json",
        fcall_name="data/input/function_calling_tests.json"
    )

    assert len(loader.fn_defs) > 0
    assert len(loader.test_prompts) > 0


def test_loader_valid_files_created(tmp_path) -> None:
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
    with pytest.raises(LoaderError):
        Loader("does_not_exist.json", "data/input/function_calling_tests.json")

# tmp path is built-in pytest variable
def test_loader_broken_json(tmp_path) -> None:
    broken_file: Path = tmp_path / "broken.json"
    broken_file.write_text("{ this is not a valid json object }")

    with pytest.raises(LoaderError):
        Loader(str(broken_file), str(broken_file))


if __name__ == "__main__":
    from pathlib import Path
    path_obj = Path(__file__)
    print (f"\n=== TESTING {path_obj.stem}.py ===\n")
    pytest.main(["-v", "-o", "python_classes=*Suite", __file__])