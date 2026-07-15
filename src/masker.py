"""Module for dynamically masking LLM vocabularies."""

from typing import Any, cast
from functools import lru_cache
import numpy as np
from .fsm import JSONState
from .llm import LLM


class TokenMasker:
    """Filters valid tokens based on JSON state and schemas."""
    def __init__(self, llm: LLM) -> None:
        """Initializes the instance."""
        self.llm = llm

        self.clean_tokens = [
            (s.replace("Ġ", " "), id) for s, id in llm.token2id.items()
        ]

        # precompute token strings and stop tokens to avoid looping
        self.token_strs = self.llm.token_strings.astype(str)
        self.stripped_strs = np.char.lstrip(self.token_strs)

        self.stop_token_ids = {
            id
            for clean, id in self.clean_tokens
            if clean.strip() in (",", "}", "]")
        }

        self.quote_ids = [
            id for clean, id in self.clean_tokens if clean.strip() == '"'
        ]

        S = JSONState
        self.state_handlers = {
            S.EXPECT_OBJECT_START: lambda p, c: self._get_tokens_for_string(
                "{", p
            ),
            S.EXPECT_ARRAY_START: lambda p, c: self._get_tokens_for_string(
                "[", p
            ),
            S.EXPECT_COLON: lambda p, c: self._get_tokens_for_string(":", p),
            S.EXPECT_KEY: lambda p, c: self._get_tokens_for_options(
                list(c["stack"][-1].remaining_keys) if c["stack"] else [], p
            ),
            S.EXPECT_COMMA_OR_END: lambda p, c: (
                self._get_tokens_for_comma_or_end(p, c)
            ),
            S.EXPECT_VALUE: lambda p, c: self._get_tokens_for_value(p, c),
        }

    def get_valid_tokens_for_state(
        self, state: JSONState, current_prefix: str, context: dict[str, Any]
    ) -> list[int]:
        """Executes get valid tokens for state."""
        handler = self.state_handlers.get(state)
        if handler:
            return handler(current_prefix, context)
        return []

    @lru_cache(maxsize=1024)
    def _get_tokens_for_string(
        self, expected: str, curr_prefix: str
    ) -> list[int]:
        """Executes get tokens for string."""
        curr_prefix = curr_prefix.lstrip()
        if curr_prefix == expected or not expected.startswith(curr_prefix):
            return []

        remainder = expected[len(curr_prefix):]

        if not curr_prefix:
            mask = np.array(
                [
                    remainder.startswith(s) and s != ""
                    for s in self.stripped_strs
                ],
                dtype=bool,
            )
        else:
            mask = np.array(
                [remainder.startswith(s) and s != "" for s in self.token_strs],
                dtype=bool,
            )
        return cast(list[int], self.llm.token_ids[mask].tolist())

    def _get_tokens_for_options(
        self, options: list[str], current_prefix: str
    ) -> list[int]:
        """Executes get tokens for options."""
        valid_ids = set()
        current_prefix = current_prefix.strip()

        for expected in options:
            if expected.startswith(current_prefix):
                tokens = self._get_tokens_for_string(expected, current_prefix)
                valid_ids.update(tokens)

        return list(valid_ids)

    def _get_tokens_for_comma_or_end(
        self, current_prefix: str, context: dict[str, Any]
    ) -> list[int]:
        """Executes get tokens for comma or end."""
        stack = context["stack"]
        if not stack:
            return []

        current_node = stack[-1]
        options = []
        if current_node.type == "object":
            if not current_node.remaining_keys:
                options.append("}")
            else:
                options.append(",")
        else:
            options.append("]")
            # For array, we could check if items are bounded,
            # but usually they are open-ended list of items.
            options.append(",")

        return self._get_tokens_for_options(options, current_prefix)

    @lru_cache(maxsize=1024)
    def _get_string_tokens(self) -> list[int]:
        # Ban tokens containing quote to prevent FSM spillover
        """Executes get string tokens."""
        no_quote_mask = np.char.find(self.token_strs, '"') == -1
        # Explicitly allow exact quote to close the string
        exact_quote_mask = np.char.strip(self.token_strs) == '"'

        no_newline_mask = (np.char.find(self.token_strs, "\n") == -1) & (
            np.char.find(self.token_strs, "Ċ") == -1
        )
        nonempty_mask = self.token_strs != ""

        valid_mask = (
            (no_quote_mask | exact_quote_mask)
            & no_newline_mask
            & nonempty_mask
        )
        return cast(list[int], self.llm.token_ids[valid_mask].tolist())

    @lru_cache(maxsize=1024)
    def _get_number_tokens(
        self, is_empty_prefix: bool, has_digits: bool, allowed_chars: str
    ) -> list[int]:
        """Executes get number tokens."""
        def is_valid_num(s: str) -> bool:
            """Executes is valid num."""
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
        mask = vec_is_valid(self.token_strs)
        return cast(list[int], self.llm.token_ids[mask].tolist())

    def _get_tokens_for_value(
        self, current_prefix: str, context: dict[str, Any]
    ) -> list[int]:
        """Executes get tokens for value."""
        current_node = context["stack"][-1]

        # get the schema for the current value we are parsing
        val_type = current_node.get_child_type(context.get("current_key"))

        if val_type == "string":
            if current_prefix.strip() == "":
                return self._get_tokens_for_string('"', current_prefix)
            return self._get_string_tokens()

        elif val_type == "enum":
            if context.get("current_key") == '"name"':
                options = context["allowed_funcs"]
                return self._get_tokens_for_options(options, current_prefix)
            else:
                if current_prefix.strip() == "":
                    return self._get_tokens_for_string('"', current_prefix)
                return self._get_string_tokens()

        elif val_type in ("number", "integer"):
            if current_node.type == "object":
                if not current_node.remaining_keys:
                    allowed_chars = "}"
                else:
                    allowed_chars = ","
            else:
                allowed_chars = ",]"

            return self._get_number_tokens(
                current_prefix.strip() == "",
                any(c.isdigit() for c in current_prefix),
                allowed_chars,
            )

        elif val_type in ("boolean", "bool"):
            return self._get_tokens_for_options(
                ["true", "false"], current_prefix
            )

        return []
