import importlib
from typing import Any
import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr


class LLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Public fields (validated by pydantic)
    llm_path: str
    llm_name: str
    hf_model: str | None = None
    use_tokenizer: bool = False

    # Private attrs (internal state, not validated)
    # to get access to them we need to use @property
    _model: Any = PrivateAttr(default=None)
    _custom_tokenizer: Any = PrivateAttr(default=None)
    _token2id: dict[str, int] = PrivateAttr(default_factory=dict)
    _id2token: dict[int, str] = PrivateAttr(default_factory=dict)
    _token_strings: Any = PrivateAttr(default=None)  # np.ndarray
    _token_ids: Any = PrivateAttr(default=None)  # np.ndarray

    def model_post_init(self, __context: Any) -> None:
        """Loads the LLM and vocabulary after Pydantic validation."""
        self._init_llm()
        if self.use_tokenizer:
            from .tokenizer import BPETokenizer

            try:
                self._custom_tokenizer = BPETokenizer(
                    vocab_file=self._model.get_path_to_vocab_file(),
                    merges_file=self._model.get_path_to_merges_file(),
                )
            except Exception as e:
                print(f"Warning: Failed to load custom tokenizer files ({e}). Falling back to HuggingFace tokenizer.")
                self.use_tokenizer = False
        self._load_vocab()

    def _init_llm(self) -> None:
        """Dynamically imports and instantiates the language model class."""
        module = importlib.import_module(self.llm_path)
        model_class = getattr(module, self.llm_name)
        if self.hf_model:
            self._model = model_class(model_name=self.hf_model)
        else:
            self._model = model_class()

    def _load_vocab(self) -> None:
        """Extracts the vocabulary mapping from the loaded model's
        tokenizer."""
        if self.use_tokenizer:
            self._token2id = self._custom_tokenizer.vocab
            self._id2token = self._custom_tokenizer.vocab_rev
        else:
            self._token2id = self._model._tokenizer.get_vocab()
            self._id2token = {
                v: k for k, v in self._token2id.items()}
        # sorting by id to ensure stable and deterministic ordering
        sorted_items = sorted(self._token2id.items(), key=lambda x: x[1])
        # replace weird G with space to save time in the decoder loop
        clean_strings = [s.replace("\u0120", " ") for s, _ in sorted_items]
        ids = [i for _, i in sorted_items]
        self._token_strings = np.array(clean_strings, dtype=object)
        self._token_ids = np.array(ids, dtype=np.int32)

    # Property accessors
    @property
    def model(self) -> Any:
        return self._model

    @property
    def token2id(self) -> dict[str, int]:
        return self._token2id

    @property
    def id2token(self) -> dict[int, str]:
        return self._id2token

    @property
    def token_strings(self) -> Any:
        return self._token_strings

    @property
    def token_ids(self) -> Any:
        return self._token_ids

    # Public methods
    def get_vocab_size(self) -> int:
        """Returns the total number of tokens in the model's vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return len(self._token2id)

    def get_logits(self, input_ids: list[int]) -> list[float] | Any:
        """Calculates the next-token logits given a sequence of input IDs.

        Args:
            input_ids (list[int]): The context sequence of token IDs.

        Returns:
            list[float] | Any: The raw logit scores for the next token
            prediction.
        """
        return self._model.get_logits_from_input_ids(input_ids)

    def encode(
        self, text: str, apply_chat_template: bool = True
    ) -> list[int] | Any:
        """Encodes text into token IDs, optionally applying chat templates.

        Args:
            text (str): The raw string prompt to encode.
            apply_chat_template (bool): Whether to wrap text in chat tags.

        Returns:
            list[int] | Any: A flat list of integer token IDs.
        """
        if apply_chat_template:
            if self.use_tokenizer:
                prompt_ids = self._custom_tokenizer.encode(text)
                return (
                    [151644, 872, 198]
                    + prompt_ids
                    + [151645, 198, 151644, 77091, 198]
                )
            tokenizer = self._model._tokenizer
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                messages = [{"role": "user", "content": text}]
                res = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                if hasattr(res, "get") and res.get("input_ids") is not None:
                    res = res["input_ids"]
                if hasattr(res, "tolist"):
                    res = res.tolist()
                if (
                    isinstance(res, list)
                    and len(res) > 0
                    and isinstance(res[0], list)
                ):
                    res = res[0]
                return res

        if self.use_tokenizer:
            return self._custom_tokenizer.encode(text)
        res = self._model.encode(text).squeeze().tolist()
        if isinstance(res, int):
            return [res]
        return res

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of token IDs into a text string.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.

        Returns:
            str: The decoded text string.
        """
        if self.use_tokenizer:
            return str(self._custom_tokenizer.decode(tokens))
        return str(self._model.decode(tokens))
