from typing import Any, cast
from functools import lru_cache
import numpy as np
from .fsm import JSONState
from .llm import LLM


class TokenMasker:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

        self.clean_tokens = [(s.replace("Ġ", " "), id)
                             for s, id in llm.token2id.items()]

        # precompute stop tokens to avoid looping inside generate step
        self.stop_token_ids = {id for clean, id in self.clean_tokens
                               if clean.strip() in (",", "}")}

        self.state_handlers = {
            # Static strings: p ... prefix, c ... context
            JSONState.START: lambda p, c: self._get_tokens_for_string(
                "{", p),
            JSONState.NAME_KEY: lambda p, c: self._get_tokens_for_string(
                '"name"', p),
            JSONState.NAME_COLON: lambda p, c: self._get_tokens_for_string(
                ":", p),
            JSONState.COMMA_AFTER: lambda p, c: self._get_tokens_for_string(
                ",", p),
            JSONState.PARAMS_KEY: lambda p, c: self._get_tokens_for_string(
                '"parameters"', p),
            JSONState.PARAMS_COLON: lambda p, c: self._get_tokens_for_string(
                ":", p),
            JSONState.PARAMS_START: lambda p, c: self._get_tokens_for_string(
                "{", p),
            JSONState.END: lambda p, c: self._get_tokens_for_string(
                "}", p),

            # Dynamic options (uses the ctx dictionary)
            JSONState.NAME_VALUE: lambda p, c: self._get_tokens_for_options(
                c['allowed_funcs'], p),
            JSONState.PARAM_KEY: lambda p, c: self._get_tokens_for_options(
                c['allowed_params'], p),
            JSONState.PARAM_COLON: lambda p, c: self._get_tokens_for_string(
                ":", p),

            # If no parameters left, ban comma and force }
            JSONState.PARAM_NEXT: lambda p, c: self._get_tokens_for_options(
                ([","] if len(c['allowed_params']) > 0 else ["}"]), p
            ),

            # Type switching for param values
            JSONState.PARAM_VALUE: lambda p, c: self._get_tokens_for_value(
                p,
                c['param_types'].get(c['current_param']),
                len(c['allowed_params'])
            ),
        }

    def get_valid_tokens_for_state(self, state: JSONState, current_prefix: str,
                                   context: dict[str, Any]) -> list[int]:
        handler = self.state_handlers.get(state)
        if handler:
            return handler(current_prefix, context)
        return []

    @lru_cache(maxsize=1024)
    def _get_tokens_for_string(self, expected: str,
                               curr_prefix: str) -> list[int]:
        curr_prefix = curr_prefix.lstrip()
        if curr_prefix == expected or not expected.startswith(curr_prefix):
            return []

        remainder = expected[len(curr_prefix):]

        token_strs = self.llm.token_strings.astype(str)
        if not curr_prefix:
            stripped_strs = np.char.lstrip(token_strs)
            mask = np.array([
                remainder.startswith(s) and s != ""
                for s in stripped_strs
                ], dtype=bool)
        else:
            mask = np.array([
                remainder.startswith(s) and s != ""
                for s in token_strs
                ], dtype=bool)
        return cast(list[int], self.llm.token_ids[mask].tolist())

    def _get_tokens_for_options(self, options: list[str],
                                current_prefix: str) -> list[int]:
        valid_ids = set()
        current_prefix = current_prefix.strip()

        for expected in options:
            if expected.startswith(current_prefix):
                tokens = self._get_tokens_for_string(expected, current_prefix)
                valid_ids.update(tokens)

        return list(valid_ids)

    @lru_cache(maxsize=1024)
    def _get_string_tokens(self) -> list[int]:
        token_strs = self.llm.token_strings.astype(str)

        # Ban tokens containing quote to prevent FSM spillover
        no_quote_mask = np.char.find(token_strs, '"') == -1
        # Explicitly allow exact quote to close the string
        exact_quote_mask = np.char.strip(token_strs) == '"'

        no_newline_mask = (np.char.find(token_strs, '\n') == -1) & \
            (np.char.find(token_strs, 'Ċ') == -1)
        nonempty_mask = token_strs != ""

        valid_mask = (no_quote_mask | exact_quote_mask) & \
            no_newline_mask & nonempty_mask
        return cast(list[int], self.llm.token_ids[valid_mask].tolist())

    @lru_cache(maxsize=1024)
    def _get_number_tokens(self, is_empty_prefix: bool, has_digits: bool,
                           allowed_params_len: int) -> list[int]:
        def is_valid_num(s: str) -> bool:
            if is_empty_prefix:
                s = s.lstrip()
            if not s:
                return False
            if not has_digits and any(c in ",}" for c in s):
                return False
            if allowed_params_len == 0 and ',' in s:
                return False
            return all(c in "0123456789.-,}" for c in s) and ".." not in s

        # vectorize the validation function
        vec_is_valid = np.vectorize(is_valid_num, otypes=[bool])
        mask = vec_is_valid(self.llm.token_strings.astype(str))
        return cast(list[int], self.llm.token_ids[mask].tolist())

    def _get_tokens_for_value(self, current_prefix: str,
                              param_type: str | None,
                              allowed_params_len: int) -> list[int]:
        if param_type == "string":
            if current_prefix.strip() == "":
                return self._get_tokens_for_string('"', current_prefix)
            return self._get_string_tokens()

        elif param_type in ("number", "integer"):
            return self._get_number_tokens(
                current_prefix.strip() == "",
                any(c.isdigit() for c in current_prefix),
                allowed_params_len
            )

        elif param_type in ("boolean", "bool"):
            return self._get_tokens_for_options(
                ["true", "false"], current_prefix
            )

        return []
