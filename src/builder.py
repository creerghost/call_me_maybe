from .models import GenerationEvent, FunctionDefinition, FunctionParameter
from pydantic import BaseModel, PrivateAttr, ConfigDict
from .masker import ValueMasker
from .visualizer import Visualizer
from .llm import LLM
import numpy as np
from typing import Callable, Any


class JSONBuilder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: LLM
    masker: ValueMasker
    visualizer: Visualizer | None = None

    _context_ids: list[int] = PrivateAttr(default_factory=list)
    _generated_text: str = PrivateAttr(default="")
    _user_question: str = PrivateAttr(default="")

    def decode_function_call(
        self, fn_defs: list[FunctionDefinition],
        prompt_ids: list[int],
        user_question: str
    ) -> str:
        self._context_ids = list(prompt_ids)
        self._user_question = user_question
        self._generated_text = ""

        self._fast_forward('{"name": "', phase="structure")

        allowed_names = [f'{fn.name}' for fn in fn_defs]
        fn_name = self._decode_enum(options=allowed_names, phase="name")
        # finds first matching fn definition
        matched_fn = next(fn for fn in fn_defs if fn.name == fn_name)
        params = matched_fn.parameters

        if not params:
            self._fast_forward(text='"}', phase="structure")
            return self._generated_text

        self._fast_forward(text='", "parameters": {', phase="structure")

        self._decode_properties(properties=params)
        self._fast_forward(text='}}', phase="structure")
        return self._generated_text

    def _decode_properties(
        self,
        properties: dict[str, FunctionParameter]
    ) -> None:
        # recursive method which can handle nested jsons
        if not properties:
            return

        keys = list(properties.keys())
        for i, key in enumerate(keys):
            param_schema = properties[key]
            phase_name = f"param:{key}"

            self._fast_forward(text=f'"{key}": ', phase="structure")

            def decode_string() -> None:
                self._fast_forward('"', phase="structure")
                self._decode_str(phase=phase_name)
                self._fast_forward('"', phase="structure")
                
            def decode_enum() -> None:
                self._fast_forward('"', phase="structure")
                if param_schema.options:
                    self._decode_enum(param_schema.options, phase=phase_name)
                else:
                    self._decode_str(phase=phase_name)
                self._fast_forward('"', phase="structure")
                
            def decode_object() -> None:
                self._fast_forward('{', phase="structure")
                self._decode_properties(param_schema.properties or {})
                self._fast_forward('}', phase="structure")

            decoders: dict[str, Any] = {
                "string": decode_string,
                "boolean": lambda: self._decode_bool(phase=phase_name),
                "bool": lambda: self._decode_bool(phase=phase_name),
                "number": lambda: self._decode_number(
                    allowed_chars="}" if i == len(keys)-1 else ",",
                    phase=phase_name
                    ),
                "integer": lambda: self._decode_number(
                    allowed_chars="}" if i == len(keys)-1 else ",",
                    phase=phase_name
                    ),
                "enum": decode_enum,
                "object": decode_object
            }

            if param_schema.type not in decoders:
                raise ValueError(f"Unsupported type: {param_schema.type}")

            # calling lambda funciton
            decoders[param_schema.type]()

            # inject commas
            if i < len(keys) - 1:
                self._fast_forward(text=', ', phase="structure")

    def _fast_forward(self, text: str, phase: str) -> None:
        # A case when we don't need llm
        token_ids = self.llm.encode(text, apply_chat_template=False)
        for i in token_ids:
            self._context_ids.append(i)
            token_str = self.llm.decode([i])
            self._generated_text += token_str
            if self.visualizer:
                event = GenerationEvent(
                    user_question=self._user_question,
                    input_ids=self._context_ids,
                    source="hardcoded",
                    current_phase=phase,
                    fast_forwarded=True,
                    valid_ids=[i],
                    token_str=token_str,
                    next_token_id=i,
                    full_json_string=self._generated_text,
                    logits=None
                )
                self.visualizer.render(event)

    def _run_decode_loop(
        self,
        get_valid_ids: Callable[[str], list[int]],
        is_done: Callable[[str, str], tuple[bool, bool]],
        phase: str,
        logit_boosts: dict[str, float] | None = None
    ) -> str:
        current_value = ""
        token_count = 0

        # Precompute boosted token IDs if provided
        boost_tokens: dict[int, float] = {}
        if logit_boosts:
            for token_str_to_boost, boost_val in logit_boosts.items():
                for vocab_id, vocab_str in self.llm.id2token.items():
                    if token_str_to_boost in vocab_str:
                        boost_tokens[vocab_id] = boost_val

        while True:
            logits = self.llm.get_logits(self._context_ids)
            valid_ids = get_valid_ids(current_value)

            masked_logits = np.full(len(logits), -np.inf)
            masked_logits[valid_ids] = np.array(logits)[valid_ids]

            # Apply logit boosting
            if logit_boosts and token_count > 3:
                for vid, boost_val in boost_tokens.items():
                    if masked_logits[vid] > float("-inf"):
                        masked_logits[vid] += boost_val

            next_token_id = int(np.argmax(masked_logits))

            token_str = self.llm.decode([next_token_id])
            done, keep_token = is_done(current_value, token_str)
            if done and not keep_token:
                break

            current_value += token_str
            self._context_ids.append(next_token_id)
            self._generated_text += token_str
            token_count += 1
            if self.visualizer:
                event = GenerationEvent(
                    user_question=self._user_question,
                    input_ids=self._context_ids,
                    source="llm",
                    current_phase=phase,
                    fast_forwarded=False,
                    valid_ids=valid_ids,
                    token_str=token_str,
                    next_token_id=next_token_id,
                    full_json_string=self._generated_text,
                    logits=masked_logits.tolist()
                )
                self.visualizer.render(event)

            if done:
                break

        return current_value.strip()

    def _decode_enum(self, options: list[str], phase: str) -> str:
        return self._run_decode_loop(
            get_valid_ids=lambda val: self.masker.get_enum_tokens(
                options, val
            ),
            is_done=lambda val, latest_token: (
                (val + latest_token).strip() in options, True
            ),
            phase=phase
        )

    def _decode_str(self, phase: str) -> str:
        return self._run_decode_loop(
            get_valid_ids=lambda val: self.masker.get_string_tokens(),
            is_done=lambda val, latest_token: ('"' in latest_token, False),
            phase=phase,
            logit_boosts={'"': 5.0}
        )

    def _decode_bool(self, phase: str) -> str:
        return self._run_decode_loop(
            get_valid_ids=lambda val: self.masker.get_boolean_tokens(
                val
            ),
            is_done=lambda val, latest_token: (
                (val + latest_token).strip() in ["true", "false"], True
            ),
            phase=phase
        )

    def _decode_number(self, allowed_chars: str, phase: str) -> str:
        boosts = {c: 10.0 for c in allowed_chars}
        return self._run_decode_loop(
            get_valid_ids=lambda val: self.masker.get_number_tokens(
                is_empty_prefix=(val.strip() == ""),
                has_digits=any(c.isdigit() for c in val),
                allowed_chars=allowed_chars
            ),
            is_done=lambda val, latest_token: (
                any(c in allowed_chars for c in latest_token), False
            ),
            phase=phase,
            logit_boosts=boosts
        )
