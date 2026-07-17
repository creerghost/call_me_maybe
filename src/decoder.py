from typing import Any, Generator
from .llm import LLM
from .models import (
    FunctionDefinition,
    SchemaNode,
    FunctionParameter,
    GenerationEvent,
)
from .fsm import JSONState, JSONStateMachine
from .masker import TokenMasker

# ONLY for performace optimization: working with tensors to not have big loops
import torch


class ConstrainedDecoder:
    """State machine driven generator that intercepts LLM outputs."""

    def __init__(self, llm: LLM) -> None:
        """Initializes the instance."""
        self.llm = llm
        self.fsm = JSONStateMachine()
        self.masker = TokenMasker(llm)
        self.stop_token_ids_tensor = torch.tensor(
            list(self.masker.stop_token_ids), dtype=torch.long
        )
        self.quote_ids_tensor = torch.tensor(
            self.masker.quote_ids, dtype=torch.long
        )

    def generate(
        self,
        prompt: str,
        user_question: str,
        func_defs: list[FunctionDefinition],
    ) -> Generator[GenerationEvent, None, None]:
        """Yields events representing the state of the FSM for each token
        generated.

        Args:
            prompt (str): The initial system/user prompt sent to
                the LLM.
            user_question (str): The raw string of the user question.
            func_defs (list[FunctionDefinition]): Available function
                definitions.

        Yields:
            GenerationEvent: The event payload containing token IDs,
                context, and state.
        """
        input_ids = self.llm.encode(prompt)

        state = JSONState.EXPECT_OBJECT_START
        generated_tokens: list[int] = []
        current_prefix = ""
        full_json_string = ""
        state_token_count = 0

        # start by expecting the root JSON object which has two keys
        root_node = SchemaNode(
            type="object",
            remaining_keys={'"name"', '"parameters"'},
            properties={
                '"name"': FunctionParameter(type="enum"),
                '"parameters"': FunctionParameter(
                    type="object", properties={}
                ),
            },
        )
        # as we parse deeper into json we will push more nodes into stack
        context: dict[str, Any] = {
            "stack": [root_node],
            "func_defs": {f'"{f.name}"': f for f in func_defs},
            "allowed_funcs": [f'"{f.name}"' for f in func_defs],
            "current_key": None,
        }

        while state != JSONState.DONE:
            valid_ids = self.masker.get_valid_tokens_for_state(
                state, current_prefix, context
            )

            if state == JSONState.EXPECT_VALUE:
                current_node = context["stack"][-1]
                val_type = current_node.get_child_type(
                    context.get("current_key")
                )
                if val_type == "string" and state_token_count > 20:
                    valid_ids = self.masker.quote_ids

            if not valid_ids:
                break

            if len(valid_ids) == 1:
                next_token_id = valid_ids[0]
                token_str = self.llm.id2token[next_token_id].replace("Ġ", " ")
                fast_forwarded = True
                logits_out = None
            else:
                next_token_id, token_str, logits_out = (
                    self._generate_token_with_llm(
                        input_ids,
                        generated_tokens,
                        valid_ids,
                        state,
                        context,
                        state_token_count,
                    )
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
            yield GenerationEvent(
                user_question=user_question,
                input_ids=input_ids,
                state=state,
                old_state=old_state,
                fast_forwarded=fast_forwarded,
                valid_ids=valid_ids,
                token_str=token_str,
                next_token_id=next_token_id,
                full_json_string=full_json_string,
                context=context,
                logits=logits_out,
            )

            if len(generated_tokens) > 200:
                break

    def _generate_token_with_llm(
        self,
        input_ids: list[int],
        generated_tokens: list[int],
        valid_ids: list[int],
        state: JSONState,
        context: dict[str, Any],
        state_token_count: int,
    ) -> tuple[int, str, list[float]]:
        """Runs the LLM forward pass, masking logits according to valid schema
        tokens.

        Args:
            input_ids (list[int]): Full list of input tokens.
            generated_tokens (list[int]): Tokens generated so far.
            valid_ids (list[int]): The tokens mathematically allowed
                next.
            state (JSONState): The current state.
            context (dict[str, Any]): JSON generation stack context.
            state_token_count (int): Token count for the current state.

        Returns:
            tuple[int, str, list[float]]: The chosen token ID, string,
                and raw logits.
        """
        logits = self.llm.get_logits(input_ids + generated_tokens)

        logits_t = torch.tensor(logits, dtype=torch.float32)
        valid_ids_tensor = torch.tensor(valid_ids, dtype=torch.long)
        mask = torch.full((len(logits_t),), float("-inf"))
        mask[valid_ids_tensor] = logits_t[valid_ids_tensor]

        if state == JSONState.EXPECT_VALUE:
            current_node = context["stack"][-1]
            val_type = current_node.get_child_type(context.get("current_key"))

            if val_type in ("number", "integer"):
                mask[self.stop_token_ids_tensor] = torch.where(
                    mask[self.stop_token_ids_tensor] > float("-inf"),
                    mask[self.stop_token_ids_tensor] + 10.0,
                    mask[self.stop_token_ids_tensor],
                )
            elif val_type == "string" and state_token_count > 3:
                mask[self.quote_ids_tensor] = torch.where(
                    mask[self.quote_ids_tensor] > float("-inf"),
                    mask[self.quote_ids_tensor] + 5.0,
                    mask[self.quote_ids_tensor],
                )

        next_token_id = int(torch.argmax(mask).item())
        token_str = self.llm.id2token[next_token_id].replace("Ġ", " ")

        return next_token_id, token_str, logits
