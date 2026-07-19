from pydantic import BaseModel, ConfigDict, PrivateAttr
from typing import Any, cast
from functools import lru_cache
import numpy as np
from .llm import LLM


class ValueMasker(BaseModel):
    """Filters valid tokens based on data types for constrained decoding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: LLM

    _clean_tokens: list[tuple[str, int]] = PrivateAttr(default_factory=list)
    _token_strs: Any = PrivateAttr(default_factory=list)
    _stripped_strs: Any = PrivateAttr(default_factory=list)
    _stop_token_ids: set[int] = PrivateAttr(default_factory=set)
    _quote_ids: list[int] = PrivateAttr(default_factory=list)

    def __hash__(self) -> int:
        return id(self)

    def model_post_init(self, __context: Any) -> None:
        """Precomputes token strings and masks."""
        self._clean_tokens = [
            (s.replace("Ġ", " "), id) for s, id in self.llm.token2id.items()
        ]

        self._token_strs = self.llm.token_strings.astype(str)
        self._stripped_strs = np.char.lstrip(self._token_strs)

        self._stop_token_ids = {
            id
            for clean, id in self._clean_tokens
            if clean.strip() in (",", "}", "]")
        }

        self._quote_ids = [
            id for clean, id in self._clean_tokens if clean.strip() == '"'
        ]

    @lru_cache(maxsize=1024)
    def get_tokens_for_string(
        self, expected: str, curr_prefix: str
    ) -> list[int]:
        """Finds all tokens that match a specific exact string prefix."""
        curr_prefix = curr_prefix.lstrip()
        if curr_prefix == expected or not expected.startswith(curr_prefix):
            return []

        remainder = expected[len(curr_prefix):]

        if not curr_prefix:
            mask = np.array(
                [
                    remainder.startswith(s) and s != ""
                    for s in self._stripped_strs
                ],
                dtype=bool,
            )
        else:
            mask = np.array(
                [remainder.startswith(s) and s != "" for s in self._token_strs],
                dtype=bool,
            )
        return cast(list[int], self.llm.token_ids[mask].tolist())

    def get_enum_tokens(
        self, options: list[str], current_prefix: str
    ) -> list[int]:
        """Finds tokens that match any of the provided string options."""
        valid_ids = set()
        current_prefix = current_prefix.strip()

        for expected in options:
            if expected.startswith(current_prefix):
                tokens = self.get_tokens_for_string(expected, current_prefix)
                valid_ids.update(tokens)

        return list(valid_ids)

    def get_boolean_tokens(self, current_prefix: str) -> list[int]:
        """Finds tokens valid for a boolean value."""
        return self.get_enum_tokens(["true", "false"], current_prefix)

    @lru_cache(maxsize=1024)
    def get_string_tokens(self) -> list[int]:
        """Computes valid tokens for a JSON string value (no quotes)."""
        no_quote_mask = np.char.find(self._token_strs, '"') == -1
        exact_quote_mask = np.char.strip(self._token_strs) == '"'

        no_newline_mask = (np.char.find(self._token_strs, "\n") == -1) & (
            np.char.find(self._token_strs, "Ċ") == -1
        )
        nonempty_mask = self._token_strs != ""

        valid_mask = (
            (no_quote_mask | exact_quote_mask)
            & no_newline_mask
            & nonempty_mask
        )
        return cast(list[int], self.llm.token_ids[valid_mask].tolist())

    @lru_cache(maxsize=1024)
    def get_number_tokens(
        self, is_empty_prefix: bool, has_digits: bool, allowed_chars: str
    ) -> list[int]:
        """Computes valid tokens for a JSON number value."""
        def is_valid_num(s: str) -> bool:
            if is_empty_prefix:
                s = s.lstrip()
            if not s:
                return False
            if not has_digits and any(c in allowed_chars for c in s):
                return False
            return (
                all(c in f"0123456789.-{allowed_chars}" for c in s)
                and ".." not in s
            )

        vec_is_valid = np.vectorize(is_valid_num, otypes=[bool])
        mask = vec_is_valid(self._token_strs)
        return cast(list[int], self.llm.token_ids[mask].tolist())
