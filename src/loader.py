import json
from pydantic import ValidationError, BaseModel, PrivateAttr, ConfigDict
from .models import FunctionDefinition, TestPrompt
from .catch import LoaderError
from typing import Any


class Loader(BaseModel):
    """Loads and validates JSON definition and prompt files.

    Attributes:
        fdef_name (str): Path to the functions definition JSON file.
        fcall_name (str | None): Path to the prompts JSON file.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fdef_name: str
    fcall_name: str | None = None

    _fn_defs: list[FunctionDefinition] = PrivateAttr(default_factory=list)
    _test_prompts: list[TestPrompt] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Initializes the loader by loading the JSON files into memory.

        Args:
            __context (Any): Context for pydantic post-init.
        """
        self._load()

    def _load(self) -> None:
        """Loads and parses JSON files into Pydantic models.

        Raises:
            LoaderError: If files are missing, empty, or contain invalid JSON.
        """
        try:
            with open(self.fdef_name, "r") as f:
                json_defs = json.load(f)
            self._fn_defs = [FunctionDefinition(**d) for d in json_defs]
            if not self._fn_defs:
                raise LoaderError("Json is empty.")

        except FileNotFoundError:
            raise LoaderError(f"File not found: {self.fdef_name}")
        except json.JSONDecodeError as e:
            raise LoaderError(f"Invalid JSON in {self.fdef_name}: {e}")
        except ValidationError as e:
            raise LoaderError(f"Schema mismatch in {self.fdef_name}:\n{e}")

        if self.fcall_name:
            try:
                with open(self.fcall_name, "r") as f:
                    json_prompts = json.load(f)
                self._test_prompts = [TestPrompt(**p) for p in json_prompts]
                if not self._test_prompts:
                    raise LoaderError("Json is empty.")
            except FileNotFoundError:
                raise LoaderError(f"File not found: {self.fcall_name}")
            except json.JSONDecodeError as e:
                raise LoaderError(f"Invalid JSON in {self.fcall_name}: {e}")
            except ValidationError as e:
                raise LoaderError(f"Schema mismatch in "
                                  f"{self.fcall_name}:\n{e}")

    @property
    def fn_defs(self) -> list[FunctionDefinition]:
        """Returns the loaded function definitions.

        Returns:
            list[FunctionDefinition]: A list of parsed function schemas.
        """
        return self._fn_defs

    @property
    def test_prompts(self) -> list[TestPrompt]:
        """Returns the loaded test prompts.

        Returns:
            list[TestPrompt]: A list of parsed test prompts.
        """
        return self._test_prompts
