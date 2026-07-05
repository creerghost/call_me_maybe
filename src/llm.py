import importlib
import json
from typing import Any


class LLM():
    def __init__(self, llm_path: str, llm_name: str,
                 hf_model: str | None = None) -> None:
        self.llm_path = llm_path
        self.llm_name = llm_name
        self.hf_model = hf_model
        self._init_llm()
        self._load_vocab()

    def _init_llm(self) -> None:
        module = importlib.import_module(self.llm_path)
        model_class = getattr(module, self.llm_name)
        if self.hf_model:
            self.model = model_class(model_name=self.hf_model)
        else:
            self.model = model_class()

    def _load_vocab(self) -> None:
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r") as f:
            self.token2id = json.load(f)
        self.id2token = {v: k for k, v in self.token2id.items()}

    def get_vocab_size(self) -> int:
        return len(self.token2id)

    def get_logits(self, input_ids: list[int]) -> list[float] | Any:
        return self.model.get_logits_from_input_ids(input_ids)

    def encode(self, text: str) -> list[int] | Any:
        return self.model.encode(text).squeeze().tolist()
