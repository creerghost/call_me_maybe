from enum import Enum, auto
from typing import Any
from .llm import LLM
from .models import FunctionDefinition
from functools import lru_cache


class JSONState(Enum):
    """
    Thinks of states as "What are we expecting the LLM to write next?
    START: We are at the very beginning. We expect '{'
    NAME_KEY: We expect 'name', etc.

    Each auto() assigns an unique integer to the state.
    By looking at state, our code knows exactly which part of the JSON
        object it is currently forcing the LLM to write.
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
        self.clean_tokens = [(s.replace("Ġ", " "), id)
                             for s, id in llm.token2id.items()]
        # variable for logit boosting (before I tried to find ',' or '}'
        # over 150,000 times in self.clean_tokens)
        # instead of looping on every num generation step, it's checking in set
        self.stop_token_ids = {id for clean, id in self.clean_tokens
                               if clean.strip() in (",", "}")}
        self.state_handlers = {
            # Static strings: p ... prefix, c ... context
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
            # If not parameters left, banningthe comma from being generated
            # and forcing to write } instead.
            JSONState.PARAM_NEXT: lambda p, c: self._get_tokens_for_options(
                ([","] if len(c['allowed_params']) > 0 else []) + ["}"], p
            ),
            # Unpacking the content so value function will becoma hashable
            JSONState.PARAM_VALUE: lambda p, c: self._get_tokens_for_value(
                p,
                c['param_types'].get(c['current_param']),
                len(c['allowed_params'])
            ),
        }

    def get_valid_tokens_for_state(self, state: JSONState, current_prefix: str,
                                   context: dict[str, Any]) -> list[int]:
        """
        Checks if current_prefix is finished and we should move to the
            next state.

        handler: correct lambda function in self.state_handlers which is
            executed.
        """
        handler = self.state_handlers.get(state)

        if handler:
            return handler(current_prefix, context)
        return []

    @lru_cache(maxsize=1024)
    def _get_tokens_for_string(self, expected: str,
                               curr_prefix: str) -> list[int]:
        """
        When calling this method, we ask to find tokens that make, for
            example, string "{".
        """
        valid_ids: list[int] = []
        # cleans LLM generated whitespace trails
        curr_prefix = curr_prefix.lstrip()
        # safe checks
        if curr_prefix == expected:
            return []

        if not expected.startswith(curr_prefix):
            return []
        # if expected="name" and prefix is "na", the remainder will be "me"
        remainder = expected[len(curr_prefix):]

        for clean_str, token_id in self.clean_tokens:
            # edge case for very first iterations: empty string
            if not curr_prefix:
                clean_str = clean_str.lstrip()

            if not clean_str:  # if token is whitespace, we allow it
                continue
            # if the remainder is "me" and token is "m", we allow it
            if remainder.startswith(clean_str):
                valid_ids.append(token_id)
        return valid_ids

    def _get_tokens_for_options(self, options: list[str],
                                current_prefix: str) -> list[int]:
        """
        Looks at the entire x-word vocabulary of the LLM.
        If it expects 'name' and have already generated 'n',
        this method returns the token IDs for 'ame', 'am', 'a', etc.

        It additionaly checks against a list of possible strings:
            options: list of available function names (for example, "fn_greet")
        """
        valid_ids = set()  # pythonic way to remove duplicates
        current_prefix = current_prefix.strip()

        for expected in options:
            if expected.startswith(current_prefix):
                tokens = self._get_tokens_for_string(expected, current_prefix)
                valid_ids.update(tokens)

        return list(valid_ids)

    @lru_cache(maxsize=1024)
    def _get_string_tokens(self) -> list[int]:
        valid: list[int] = []
        for clean, id in self.clean_tokens:
            if clean.strip() == '"':
                valid.append(id)
            elif '"' not in clean:
                valid.append(id)
        return valid

    @lru_cache(maxsize=1024)
    def _get_number_tokens(self, is_empty_prefix: bool, has_digits: bool,
                           allowed_params_len: int) -> list[int]:
        valid: list[int] = []
        for clean, id in self.clean_tokens:
            if is_empty_prefix:
                clean = clean.lstrip()
            if not clean:
                continue
            if not has_digits and any(c in ",}" for c in clean):
                continue
            if allowed_params_len == 0 and ',' in clean:
                continue
            if all(c in "0123456789.-,}"
                   for c in clean) and ".." not in clean:
                valid.append(id)
        return valid

    def _get_tokens_for_value(self, current_prefix: str,
                              param_type: str | None,
                              allowed_params_len: int) -> list[int]:
        """Return valid tokens for the current parameter value type."""
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

        elif param_type == "boolean":
            return self._get_tokens_for_options(
                ["true", "false"], current_prefix
            )

        return []

    def generate(self, prompt: str, user_question: str,
                 func_defs: list[FunctionDefinition],
                 visualize: bool = False) -> Any | str:
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
        # user prompt is turned into numbers
        input_ids = self.llm.encode(prompt)
        state = JSONState.START
        generated_tokens: list[int] = []
        current_prefix = ""
        full_json_string = ""
        # context holds dynamic knowledge. We map function names to their
        # actual definitions so we can look up their parameters later
        context = {
            'func_defs': {f'"{f.name}"': f for f in func_defs},
            'allowed_funcs': [f'"{f.name}"' for f in func_defs],
            'allowed_params': [],
            'param_types': {},
            'current_param': None
        }

        while state != JSONState.DONE:
            valid_ids = self.get_valid_tokens_for_state(state, current_prefix,
                                                        context)
            if not valid_ids:
                print(f"FATAL: No valid tokens for state{state.name} "
                      f"with prefix '{current_prefix}'. Breaking.")
                break

            # fast forward optimization: if only 1 token is valid,
            # completely skip the expensive LLM forward pass and
            # force the token immediately.
            if len(valid_ids) == 1:
                next_token_id = valid_ids[0]
                token_str = self.llm.id2token[next_token_id].replace("Ġ", " ")
                fast_forwarded = True
                if not visualize:
                    print(f"[GENERATE] state={state.name}, valid_tokens=1, "
                          f"chosen='{token_str}' "
                          f"(fast-forwarded - optimization)")
            else:
                fast_forwarded = False
                # if multiple valid tokens exist, we must ask the LLM
                logits = self.llm.get_logits(input_ids + generated_tokens)

                # C-styled array masking
                # before I looped 150,000 times checking if i not in valid_set
                # now I have 1 list of -inf floats and I need to iterate over
                # 5 or 6 valid tokens
                new_logits = [float("-inf")] * len(logits)
                for vid in valid_ids:
                    new_logits[vid] = logits[vid]
                logits = new_logits

                # logit boosting: if we are writing a number, boost the
                # probability of ',' and '}'
                if state == JSONState.PARAM_VALUE:
                    param_type = context['param_types'].get(  # type: ignore
                        context['current_param'])
                    if param_type in ("number", "integer"):
                        for id in self.stop_token_ids:
                            if logits[id] > float("-inf"):
                                logits[id] += 10.0

                # take the token with maximum probability
                next_token_id = logits.index(max(logits))
                token_str = self.llm.id2token[next_token_id].replace("Ġ", " ")
                if not visualize:
                    print(f"[GENERATE] state={state.name}, "
                          f"valid_tokens={len(valid_ids)}, "
                          f"chosen='{token_str}'")

            generated_tokens.append(next_token_id)
            current_prefix += token_str
            full_json_string += token_str

            old_state = state
            state, current_prefix = self._transition_state(state,
                                                           current_prefix,
                                                           context)

            if visualize:
                self._render_dashboard(user_question, state, old_state,
                                       fast_forwarded, valid_ids, token_str,
                                       full_json_string)
            else:
                if old_state != state:
                    print(f"[GENERATE] transitioned to {state.name}\n")
                else:
                    print(f"[GENERATE] Staying in the same state "
                          f"{old_state.name}. "
                          f"({old_state.name} == {state.name})")

            if len(generated_tokens) > 200:
                break

        return self.llm.model.decode(generated_tokens)

    def _transition_state(self, state: JSONState,
                          current_prefix: str,
                          context: dict[str, Any]) -> tuple[JSONState, str]:
        """
        Once current_prefix exactly matches what we were expecting, we wipe
            current_prefix clean and move to the last logical state.

        E.g., if we were in START and current_prefix is now "{", we transition
            to NAME_KEY.
        """
        # print(f"[TRANSITION_STATE] state={state.name}, "
        #       f"prefix='{current_prefix}'")

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
                func_defs = context['func_defs'][current_prefix.strip()]
                context['allowed_params'] = [
                    f'"{p}"' for p in func_defs.parameters.keys()
                ]
                context['param_types'] = {
                    f'"{p}"': v.type for p, v in func_defs.parameters.items()
                }
                return JSONState.COMMA_AFTER, ""

        elif state == JSONState.PARAM_KEY:
            if current_prefix.strip() in context['allowed_params']:
                context['current_param'] = current_prefix.strip()
                context['allowed_params'].remove(context['current_param'])
                return JSONState.PARAM_COLON, ""

        elif state == JSONState.PARAM_VALUE:
            param_type = context['param_types'].get(context['current_param'])
            if param_type == "string":
                prefix_stripped = current_prefix.strip()
                if current_prefix.endswith('"') and len(prefix_stripped) > 1:
                    return JSONState.PARAM_NEXT, ""
            elif param_type in ("number", "integer"):
                if "}" in current_prefix:
                    return JSONState.END, ""
                elif "," in current_prefix:
                    return JSONState.PARAM_KEY, ""
            elif param_type == "boolean":
                stripped = current_prefix.strip()
                if stripped in ("true", "false"):
                    return JSONState.PARAM_NEXT, ""

        elif state == JSONState.PARAM_NEXT:
            if current_prefix.strip() == ",":
                return JSONState.PARAM_KEY, ""
            elif current_prefix.strip() == "}":
                return JSONState.END, ""

        # 3. If we are not done with this state, stay in the same state
        return state, current_prefix

    def _render_dashboard(self, user_question: str, state: JSONState,
                          old_state: JSONState, fast_forwarded: bool,
                          valid_ids: list[int], token_str: str,
                          full_json_string: str) -> None:
        dashboard = "\033[2J\033[H"  # clear screen & cursor home
        dashboard += ("\033[96m=== Constrained JSON Decoder "
                      "===\033[0m\n\n")
        dashboard += f"\033[93mUser Prompt:\033[0m {user_question}\n\n"
        dashboard += (f"\033[92mCurrent State:\033[0m {state.name} "
                      f"(was {old_state.name})\n")

        if fast_forwarded:
            dashboard += ("\033[94mAllowed Tokens:\033[0m 1 "
                          "(Fast-Forwarding!)\n")
        else:
            dashboard += (f"\033[94mAllowed Tokens:\033[0m "
                          f"{len(valid_ids)}\n")

        dashboard += f"\033[95mGenerated Token:\033[0m '{token_str}'\n\n"

        colored_json = full_json_string.replace('"', '\033[36m"\033[0m')
        colored_json = colored_json.replace(':', '\033[33m:\033[0m')
        dashboard += f"{colored_json}\n"

        print(dashboard, end="", flush=True)
