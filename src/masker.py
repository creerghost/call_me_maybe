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
    _str_to_ids: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _stripped_to_ids: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _numeric_tokens: list[tuple[str, int]] = PrivateAttr(default_factory=list)

    def __hash__(self) -> int:
        """Required for lru_cache on methods since BaseModel isn't hashable.

        Returns:
            int: The object's memory address.
        """
        return id(self)

    def model_post_init(self, __context: Any) -> None:
        """Precomputes token strings and masks.

        Args:
            __context (Any): Context for pydantic post-init.
        """
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

        self._str_to_ids = {}
        self._stripped_to_ids = {}
        for clean, id in self._clean_tokens:
            if clean:
                self._str_to_ids.setdefault(clean, []).append(id)
                self._stripped_to_ids.setdefault(clean.lstrip(), []).append(id)

        valid_charset = set("0123456789.-,]} \n\r\t")
        self._numeric_tokens = [
            (clean, id) for clean, id in self._clean_tokens
            if all(c in valid_charset for c in clean)
        ]

    @lru_cache(maxsize=1024)
    def get_tokens_for_string(
        self, expected: str, curr_prefix: str
    ) -> list[int]:
        """Finds all tokens that match a specific exact string prefix.

        Args:
            expected (str): The expected string.
            curr_prefix (str): The current generated prefix.

        Returns:
            list[int]: A list of matching token IDs.
        """
        curr_prefix = curr_prefix.lstrip()
        if curr_prefix == expected or not expected.startswith(curr_prefix):
            return []

        remainder = expected[len(curr_prefix):]

        valid_ids = []
        target_dict = (self._stripped_to_ids if not curr_prefix
                       else self._str_to_ids)
        for i in range(1, len(remainder) + 1):
            prefix = remainder[:i]
            if prefix in target_dict:
                valid_ids.extend(target_dict[prefix])
        return valid_ids

    def get_enum_tokens(
        self, options: list[str], current_prefix: str
    ) -> list[int]:
        """Finds tokens that match any of the provided string options.

        Args:
            options (list[str]): The allowed string options.
            current_prefix (str): The current generated prefix.

        Returns:
            list[int]: A list of matching token IDs.
        """
        valid_ids = set()
        current_prefix = current_prefix.strip()

        for expected in options:
            if expected.startswith(current_prefix):
                tokens = self.get_tokens_for_string(expected, current_prefix)
                valid_ids.update(tokens)

        return list(valid_ids)

    def get_boolean_tokens(self, current_prefix: str) -> list[int]:
        """Finds tokens valid for a boolean value.

        Args:
            current_prefix (str): The current generated prefix.

        Returns:
            list[int]: A list of matching token IDs.
        """
        return self.get_enum_tokens(["true", "false"], current_prefix)

    @lru_cache(maxsize=1024)
    def get_string_tokens(self) -> list[int]:
        """Computes valid tokens for a JSON string value (no quotes).

        Returns:
            list[int]: A list of matching token IDs.
        """
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
        """Computes valid tokens for a JSON number value.

        Args:
            is_empty_prefix (bool): Whether the current number prefix is empty.
            has_digits (bool): Whether digits have already been generated.
            allowed_chars (str): Characters allowed after the number.

        Returns:
            list[int]: A list of matching token IDs.
        """
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

        return [
            id for s, id in self._numeric_tokens
            if is_valid_num(s)
        ]

    @lru_cache(maxsize=128)
    def get_boost_tokens(
        self, logit_boosts: tuple[tuple[str, float], ...]
    ) -> dict[int, float]:
        """Precomputes boosted logits based on specific active characters.

        Args:
            logit_boosts (tuple): A tuple of (token_str, boost_val) pairs.

        Returns:
            dict[int, float]: A mapping from token_id to boost value.
        """
        boost_tokens: dict[int, float] = {}
        for token_str_to_boost, boost_val in logit_boosts:
            mask = np.char.find(self._token_strs, token_str_to_boost) != -1
            matching_ids = self.llm.token_ids[mask]
            for vid in matching_ids:
                boost_tokens[int(vid)] = boost_val
        return boost_tokens
