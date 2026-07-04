from enum import Enum, auto
from typing import Any
from .llm import LLM
from .models import FunctionDefinition


class JSONState(Enum):
    """
    Thinks of states as "What are we expecting the LLM to write next?
    START: We are at the very beginning. We expect '{'
    NAME_KEY: We expect 'name', etc.
    """
    START = auto()         # Expecting '{'
    NAME_KEY = auto()      # Expecting '"name"'
    NAME_COLON = auto()    # Expecting ':'
    NAME_VALUE = auto()    # Expecting '"fn_add_numbers"'
    COMMA_AFTER = auto()   # Expecting ','
    PARAMS_KEY = auto()    # Expecting '"parameters"'
    PARAMS_COLON = auto()  # Expecting ':'
    PARAMS_START = auto()  # Expecting '{'
    PARAM_KEY = auto()     # Expecting '"arg1"'
    PARAM_COLON = auto()   # Expecting ':'
    PARAM_VALUE = auto()   # Expecting value
    PARAM_NEXT = auto()    # Expecting ',' or '}'
    END = auto()           # Expecting '}'
    DONE = auto()          # Finished generating


class ConstrainedDecoder:
    def __init__(self, llm: LLM) -> None:
        """
        Initalizes the LLM model.

        self.state_handlers:
            Key: The state we are in,
            Value: A small lambda function that looks up valid tokens.

        When we are in JSONState.NAME_KEY, the dictionary tells us to run:
            self._get_tokens_for_string('"name"', p). This asks the helper
            method: "Find all token IDs that can help us spell the word "name".
        """
        self.llm = llm
        self.state_handlers = {
            # Static strings
            JSONState.START: lambda p, c: self._get_tokens_for_string(
                "{", p
            ),
            JSONState.NAME_KEY: lambda p, c: self._get_tokens_for_string(
                '"name"', p
            ),
            JSONState.NAME_COLON: lambda p, c: self._get_tokens_for_string(
                ":", p
            ),
            JSONState.COMMA_AFTER: lambda p, c: self._get_tokens_for_string(
                ",", p
            ),
            JSONState.PARAMS_KEY: lambda p, c: self._get_tokens_for_string(
                '"parameters"', p
            ),
            JSONState.PARAMS_COLON: lambda p, c: self._get_tokens_for_string(
                ":", p
            ),
            JSONState.PARAMS_START: lambda p, c: self._get_tokens_for_string(
                "{", p
            ),
            JSONState.END: lambda p, c: self._get_tokens_for_string(
                "}", p
            ),

            # Dynamic options (uses the ctx dictionary)
            JSONState.NAME_VALUE: lambda p, c: self._get_tokens_for_options(
                c['allowed_funcs'], p
            ),
            JSONState.PARAM_KEY: lambda p, c: self._get_tokens_for_options(
                c['allowed_params'], p
            ),
            JSONState.PARAM_COLON: lambda p, c: self._get_tokens_for_string(
                ":", p
            ),
            JSONState.PARAM_NEXT: lambda p, c: self._get_tokens_for_options(
                [",", "}"], p
            ),
            JSONState.PARAM_VALUE: self._get_tokens_for_value,
        }

    def generate(self, prompt: str,
                 func_defs: list[FunctionDefinition]) -> Any | str:
        """
        Engine that talks to a model.

        1. logits = self.llm.get_logits() -> What do you want to say next?
        2. valid_ids = self.get_valid_tokens_for_state -> What is the model
            allowed to say next?
        3. logits[i] = float("-inf") -> Filter the all invalid tokens.
        4. next_token_id = logits.index(max(logits)) -> Take the remaining
            token with the highest probability.
        5. current_prefix += token_str -> Add the text to our current_prefix.
        6. state, current_prefix = self._transition_state() -> wipe the
            current_prefix clean and move to the next logical state.
        """
        input_ids = self.llm.encode(prompt)

        state = JSONState.START
        generated_tokens: list[int] = []
        current_prefix = ""
        context = {
            'func_defs': {f'"{f.name}"': f for f in func_defs},
            'allowed_funcs': [f'"{f.name}"' for f in func_defs],
            'allowed_params': [],
            'param_types': {},
            'current_param': None
        }

        while state != JSONState.DONE:
            logits = self.llm.get_logits(input_ids + generated_tokens)
            valid_ids = self.get_valid_tokens_for_state(state, current_prefix,
                                                        context)

            for i in range(len(logits)):
                if i not in valid_ids:
                    logits[i] = float("-inf")

            next_token_id = logits.index(max(logits))
            generated_tokens.append(next_token_id)

            token_str = self.llm.id2token[next_token_id].replace("Ġ", " ")
            current_prefix += token_str
            state, current_prefix = self._transition_state(state,
                                                           current_prefix,
                                                           context)

            if len(generated_tokens) > 200:
                break

        return self.llm.model.decode(generated_tokens)

    def get_valid_tokens_for_state(self, state: JSONState, current_prefix: str,
                                   context: dict[str, Any]) -> list[int]:
        """
        Decides what is the model allowed to say next.
        """
        handler = self.state_handlers.get(state)

        if handler:
            return handler(current_prefix, context)
        return []

    def _get_tokens_for_string(self, expected: str,
                               curr_prefix: str) -> list[int]:
        """
        Looks at the entire x-word vocabulary of the LLM.
        If it expects 'name' and have already generated 'n',
        this method returns the token IDs for 'ame', 'am', 'a', etc.
        """
        valid_ids = []
        curr_prefix = curr_prefix.strip()
        if curr_prefix == expected:
            return []
        remainder = expected[len(curr_prefix):]

        for token_str, token_id in self.llm.token2id.items():
            clean_str = token_str.replace("Ġ", " ")
            if not clean_str.strip():
                valid_ids.append(token_id)
                continue
            clean_str = clean_str.strip()
            if remainder.startswith(clean_str):
                valid_ids.append(token_id)
            elif clean_str.startswith(remainder):
                valid_ids.append(token_id)
        return valid_ids

    def _get_tokens_for_options(self, options: list[str],
                                current_prefix: str) -> list[int]:
        """
        Looks at the entire x-word vocabulary of the LLM.
        If it expects 'name' and have already generated 'n',
        this method returns the token IDs for 'ame', 'am', 'a', etc.

        It additionaly checks against a list of possible strings
        (like all allowed function names).
        """
        valid_ids = set()
        current_prefix = current_prefix.strip()

        for expected in options:
            if expected.startswith(current_prefix):
                tokens = self._get_tokens_for_string(expected, current_prefix)
                valid_ids.update(tokens)

        return list(valid_ids)

    def _get_tokens_for_value(self, current_prefix: str,
                              context: dict[str, Any]) -> list[int]:
        param_type = context['param_types'].get(context['current_param'])

        if param_type == "string":
            if current_prefix.strip() == "":
                return self._get_tokens_for_string('"', current_prefix)
            return list(self.llm.token2id.values())

        elif param_type == "number":
            valid = []
            for s, tid in self.llm.token2id.items():
                clean = s.replace("Ġ", " ").strip()
                if not clean:
                    valid.append(tid)
                    continue
                if all(c in "0123456789.-,}" for c in clean):
                    valid.append(tid)
            return valid
        return []

    def _transition_state(self, state: JSONState,
                          current_prefix: str,
                          context: dict[str, Any]) -> tuple[JSONState, str]:
        """
        Once current_prefix exactly matches what we were expecting, we wipe
            current_prefix clean and move to the last logical state.

        E.g., if we were in START and current_prefix is now "{", we transition
            to NAME_KEY.
        """
        expected_strings = {
            JSONState.START: "{",
            JSONState.NAME_KEY: '"name"',
            JSONState.NAME_COLON: ":",
            JSONState.COMMA_AFTER: ",",
            JSONState.PARAMS_KEY: '"parameters"',
            JSONState.PARAMS_COLON: ":",
            JSONState.PARAMS_START: "{",
            JSONState.PARAM_COLON: ":",
            JSONState.END: "}",
        }

        # 1. Handle Static Strings
        if state in expected_strings:
            expected = expected_strings[state]
            if current_prefix.strip() == expected:
                next_states = {
                    JSONState.START: JSONState.NAME_KEY,
                    JSONState.NAME_KEY: JSONState.NAME_COLON,
                    JSONState.NAME_COLON: JSONState.NAME_VALUE,
                    JSONState.COMMA_AFTER: JSONState.PARAMS_KEY,
                    JSONState.PARAMS_KEY: JSONState.PARAMS_COLON,
                    JSONState.PARAMS_COLON: JSONState.PARAMS_START,
                    JSONState.PARAMS_START: JSONState.PARAM_KEY,
                    JSONState.PARAM_COLON: JSONState.PARAM_VALUE,
                    JSONState.END: JSONState.DONE
                }
                return next_states[state], ""

        # 2. Handle Dynamic Options (like NAME_VALUE)
        elif state == JSONState.NAME_VALUE:
            if current_prefix.strip() in context['allowed_funcs']:
                func_def = context['func_defs'][current_prefix.strip()]
                context['allowed_params'] = [
                    f'"{p}"' for p in func_def.parameters.keys()
                ]
                context['param_types'] = {
                    f'"{p}"': v.type for p, v in func_def.parameters.items()
                }
                return JSONState.COMMA_AFTER, ""

        elif state == JSONState.PARAM_KEY:
            if current_prefix.strip() in context['allowed_params']:
                context['current_param'] = current_prefix.strip()
                return JSONState.PARAM_COLON, ""

        elif state == JSONState.PARAM_VALUE:
            param_type = context['param_types'].get(context['current_param'])
            if param_type == "string":
                prefix_stripped = current_prefix.strip()
                if current_prefix.endswith('"') and len(prefix_stripped) > 1:
                    return JSONState.PARAM_NEXT, ""
            elif param_type == "number":
                if "}" in current_prefix:
                    return JSONState.END, ""
                elif "," in current_prefix:
                    return JSONState.PARAM_KEY, ""

        elif state == JSONState.PARAM_NEXT:
            if current_prefix.strip() == ",":
                return JSONState.PARAM_KEY, ""
            elif current_prefix.strip() == "}":
                return JSONState.END, ""

        # 3. If we are not done with this state, stay in the same state
        return state, current_prefix
