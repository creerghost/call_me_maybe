import json
from .models import FunctionDefinition, TestPrompt


class Loader():
    def __init__(self, fdef_name, fcall_name) -> None:
        self.fn_defs: list[FunctionDefinition] = []
        self.test_prompts: list[TestPrompt] = []
        self._load(fdef_name, fcall_name)

    def _load(self, fdef: str, fcall: str) -> None:
        with open(fdef, "r") as f:
            json_defs = json.load(f)
        self.fn_defs = [FunctionDefinition(**d) for d in json_defs]

        with open(fcall, "r") as f:
            json_prompts = json.load(f)
        self.test_prompts = [TestPrompt(**p) for p in json_prompts]
