import importlib
from typing import Any
import numpy as np


class LLM():
    def __init__(self, llm_path: str, llm_name: str,
                 hf_model: str | None = None) -> None:
        """Initializes the LLM wrapper and loads the model and vocabulary.

        Args:
            llm_path (str): The import path of the language model module.
            llm_name (str): The class name of the language model to
            instantiate.
            hf_model (str | None): Optional HuggingFace model identifier to
            load dynamically.
        """
        self.llm_path = llm_path
        self.llm_name = llm_name
        self.hf_model = hf_model
        self._init_llm()
        self._load_vocab()

    def _init_llm(self) -> None:
        """Dynamically imports and instantiates the language model class."""
        module = importlib.import_module(self.llm_path)
        model_class = getattr(module, self.llm_name)
        if self.hf_model:
            self.model = model_class(model_name=self.hf_model)
        else:
            self.model = model_class()

    def _load_vocab(self) -> None:
        """Extracts the vocabulary mapping from the loaded model's
        tokenizer."""
        self.token2id: dict[str, int] = self.model._tokenizer.get_vocab()
        self.id2token: dict[int, str] = {v: k
                                         for k, v in self.token2id.items()}
        # sorting by id to ensure stable and deterministic ordering
        sorted_items = sorted(self.token2id.items(), key=lambda x: x[1])
        # replace weird G with space to save time in the decoder loop
        clean_strings = [s.replace("\u0120", " ") for s, _ in sorted_items]
        ids = [i for _, i in sorted_items]
        self.token_strings = np.array(clean_strings, dtype=object)
        self.token_ids = np.array(ids, dtype=np.int32)

    def get_vocab_size(self) -> int:
        """Returns the total number of tokens in the model's vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return len(self.token2id)

    def get_logits(self, input_ids: list[int]) -> list[float] | Any:
        """Calculates the next-token logits given a sequence of input IDs.

        Args:
            input_ids (list[int]): The context sequence of token IDs.

        Returns:
            list[float] | Any: The raw logit scores for the next token
            prediction.
        """
        return self.model.get_logits_from_input_ids(input_ids)

    def encode(self, text: str) -> list[int] | Any:
        """Encodes text into token IDs, automatically applying chat templates
        if available.

        Args:
            text (str): The raw string prompt to encode.

        Returns:
            list[int] | Any: A flat list of integer token IDs.
        """
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
            if isinstance(res, list) and len(
                    res) > 0 and isinstance(res[0], list):
                res = res[0]
            return res
        return self.model.encode(text).squeeze().tolist()
