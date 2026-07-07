import importlib
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
        self.token2id = self.model._tokenizer.get_vocab()
        self.id2token = {v: k for k, v in self.token2id.items()}

    def get_vocab_size(self) -> int:
        return len(self.token2id)

    def get_logits(self, input_ids: list[int]) -> list[float] | Any:
        return self.model.get_logits_from_input_ids(input_ids)

    def encode(self, text: str) -> list[int] | Any:
        tokenizer = self.model._tokenizer
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            messages = [{"role": "user", "content": text}]
            res = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            if hasattr(res, "get") and res.get("input_ids") is not None:
                res = res["input_ids"]
            if hasattr(res, "tolist"):
                res = res.tolist()
            if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                res = res[0]
            return res
        return self.model.encode(text).squeeze().tolist()
