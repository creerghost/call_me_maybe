import json
from pydantic import ValidationError
from .models import FunctionDefinition, TestPrompt
from .catch import LoaderError


class Loader:
    def __init__(self, fdef_name: str, fcall_name: str) -> None:
        """Initializes the Loader and parses the configuration files.

        Args:
            fdef_name (str): Path to the function definitions JSON file.
            fcall_name (str): Path to the test prompts JSON file.
        """
        self.fn_defs: list[FunctionDefinition] = []
        self.test_prompts: list[TestPrompt] = []
        self._load(fdef_name, fcall_name)

    def _load(self, fdef: str, fcall: str) -> None:
        """Loads, parses, and validates the input JSON files.

        Args:
            fdef (str): Path to the function definitions JSON file.
            fcall (str): Path to the test prompts JSON file.

        Raises:
            LoaderError: If files are missing, malformed, or fail Pydantic
            schema validation.
        """
        try:
            with open(fdef, "r") as f:
                json_defs = json.load(f)
            self.fn_defs = [FunctionDefinition(**d) for d in json_defs]
            if not self.fn_defs:
                raise LoaderError("Json is empty.")

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
            if not self.test_prompts:
                raise LoaderError("Json is empty.")
        except FileNotFoundError:
            raise LoaderError(f"File not found: {fcall}")
        except json.JSONDecodeError as e:
            raise LoaderError(f"Invalid JSON in {fcall}: {e}")
        except ValidationError as e:
            raise LoaderError(f"Schema mismatch in {fcall}:\n{e}")
