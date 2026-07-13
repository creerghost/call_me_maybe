from typing import Any
from .llm import LLM
from .models import FunctionDefinition
from .fsm import JSONState, JSONStateMachine
from .masker import TokenMasker
# ONLY for performace optimization: working with tensors to not have big loops
import torch


class ConstrainedDecoder:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.fsm = JSONStateMachine()
        self.masker = TokenMasker(llm)

    def generate(self, prompt: str, user_question: str,
                 func_defs: list[FunctionDefinition],
                 visualize: bool = False) -> Any | str:
        input_ids = self.llm.encode(prompt)

        state = JSONState.START
        generated_tokens: list[int] = []
        current_prefix = ""
        full_json_string = ""
        state_token_count = 0

        context = {
            'func_defs': {f'"{f.name}"': f for f in func_defs},
            'allowed_funcs': [f'"{f.name}"' for f in func_defs],
            'allowed_params': [],
            'param_types': {},
            'current_param': None
        }

        while state != JSONState.DONE:
            valid_ids = self.masker.get_valid_tokens_for_state(
                state, current_prefix, context
            )

            valid_ids = self._handle_error_recovery(
                state, context, state_token_count, valid_ids
            )

            if not valid_ids:
                print(f"FATAL: No valid tokens for state{state.name} "
                      f"with prefix '{current_prefix}'. Breaking.")
                break

            if len(valid_ids) == 1:
                next_token_id, token_str = self._fast_forward_token(
                    valid_ids, state, visualize
                )
                fast_forwarded = True
            else:
                next_token_id, token_str = self._generate_token_with_llm(
                    input_ids, generated_tokens, valid_ids,
                    state, context, state_token_count, visualize
                )
                fast_forwarded = False

            generated_tokens.append(next_token_id)
            current_prefix += token_str
            full_json_string += token_str

            old_state = state
            state, current_prefix = self.fsm.transition_state(
                state, current_prefix, context
            )

            state_token_count += 1
            if old_state != state:
                state_token_count = 0

            self._print_status(
                visualize, user_question, state, old_state,
                fast_forwarded, valid_ids, token_str, full_json_string
            )

            if len(generated_tokens) > 200:
                break

        return self.llm.model.decode(generated_tokens)

    def _handle_error_recovery(self, state: JSONState, context: dict[str, Any],
                               state_token_count: int,
                               valid_ids: list[int]) -> list[int]:
        param_types = context.get('param_types', {})
        current_param = context.get('current_param')
        if state == JSONState.PARAM_VALUE and \
                isinstance(param_types, dict) and \
                isinstance(current_param, str) and \
                param_types.get(current_param) == "string":
            if state_token_count > 20:
                print("\n[ERROR RECOVERY] String length exceeded limit."
                      " Forcing closing quote.")
                return [id for clean, id in self.masker.clean_tokens
                        if clean.strip() == '"']
        return valid_ids

    def _fast_forward_token(self, valid_ids: list[int], state: JSONState,
                            visualize: bool) -> tuple[int, str]:
        next_token_id = valid_ids[0]
        token_str = self.llm.id2token[next_token_id].replace("Ġ", " ")
        return next_token_id, token_str

    def _generate_token_with_llm(self, input_ids: list[int],
                                 generated_tokens: list[int],
                                 valid_ids: list[int], state: JSONState,
                                 context: dict[str, Any],
                                 state_token_count: int,
                                 visualize: bool) -> tuple[int, str]:
        logits = self.llm.get_logits(input_ids + generated_tokens)

        logits_t = torch.tensor(logits, dtype=torch.float32)
        valid_ids_tensor = torch.tensor(valid_ids, dtype=torch.long)
        mask = torch.full((len(logits_t),), float("-inf"))
        mask[valid_ids_tensor] = logits_t[valid_ids_tensor]

        if state == JSONState.PARAM_VALUE:
            param_type = context['param_types'].get(context['current_param'])
            if param_type in ("number", "integer"):
                stop_ids = torch.tensor(
                    list(self.masker.stop_token_ids),
                    dtype=torch.long
                )
                mask[stop_ids] = torch.where(
                    mask[stop_ids] > float("-inf"),
                    mask[stop_ids] + 10.0,
                    mask[stop_ids]
                )
            elif param_type == "string" and state_token_count > 3:
                quote_ids = [id for clean, id in
                             self.masker.clean_tokens
                             if clean.strip() == '"']
                q_ids_tensor = torch.tensor(
                    quote_ids, dtype=torch.long
                )
                mask[q_ids_tensor] = torch.where(
                    mask[q_ids_tensor] > float("-inf"),
                    mask[q_ids_tensor] + 5.0,
                    mask[q_ids_tensor]
                )

        next_token_id = int(torch.argmax(mask).item())
        token_str = self.llm.id2token[next_token_id].replace("Ġ", " ")

        return next_token_id, token_str

    def _print_status(self, visualize: bool, user_question: str,
                      state: JSONState, old_state: JSONState,
                      fast_forwarded: bool, valid_ids: list[int],
                      token_str: str, full_json_string: str) -> None:
        if visualize:
            self._render_dashboard(user_question, state, old_state,
                                   fast_forwarded, valid_ids, token_str,
                                   full_json_string)

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
