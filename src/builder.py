import re
from typing import Callable, Any, Optional

from pydantic import BaseModel, PrivateAttr, ConfigDict
import numpy as np

from .models import GenerationEvent, FunctionDefinition, FunctionParameter
from .masker import ValueMasker
from .visualizer import Visualizer
from .llm import LLM


class JSONBuilder(BaseModel):
    """Orchestrates token-by-token generation using constrained decoding."""
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
        """Executes the constrained generation loop for a user prompt.

        Args:
            fn_defs (list[FunctionDefinition]): Available function schemas.
            prompt_ids (list[int]): Tokenized input IDs of the prompt.
            user_question (str): The raw user question string.

        Returns:
            str: The final generated JSON string for the function call.
        """
        self._context_ids = list(prompt_ids)
        self._user_question = user_question
        self._generated_text = ""

        self._fast_forward('{"name": "', phase="structure")

        allowed_names = [f'{fn.name}' for fn in fn_defs]
        fn_name = self._decode_enum(options=allowed_names, phase="name")
        # finds first matching fn definition
        matched_fn = next(fn for fn in fn_defs if fn.name == fn_name)
        params = matched_fn.parameters

        self._fast_forward(text='", "parameters": {', phase="structure")

        self._decode_properties(properties=params)
        self._fast_forward(text='}}', phase="structure")
        return self._generated_text

    def _decode_properties(
        self,
        properties: dict[str, FunctionParameter]
    ) -> None:
        """Recursively decodes a dictionary of function parameters.

        Args:
            properties (dict[str, FunctionParameter]): Parameter schema dict.
        """
        if not properties:
            return

        keys = list(properties.keys())
        for i, key in enumerate(keys):
            param_schema = properties[key]
            phase_name = f"param:{key}"

            self._fast_forward(text=f'"{key}": ', phase="structure")

            def get_candidates(param_type: str) -> list[str]:
                """Extracts plausible dynamic candidates from user question.

                Args:
                    param_type (str): The type of the parameter.

                Returns:
                    list[str]: Extracted candidate substrings.
                """
                if param_type in ["number", "integer"]:
                    cands = re.findall(
                        r'\b\d+(?:\.\d+)?\b', self._user_question
                    )
                else:
                    cands = re.findall(
                        r'"([^"]*)"|\'([^\']*)\'', self._user_question
                    )
                    cands = [c[0] or c[1] for c in cands]
                return list(dict.fromkeys(cands))

            def decode_string() -> None:
                """Decodes a general string parameter."""
                self._fast_forward('"', phase="structure")
                self._decode_str(
                    phase=phase_name,
                    autocomplete_options=get_candidates(param_schema.type)
                )
                self._fast_forward('"', phase="structure")

            def decode_enum() -> None:
                """Decodes an enum string parameter."""
                self._fast_forward('"', phase="structure")
                if param_schema.options:
                    self._decode_enum(param_schema.options, phase=phase_name)
                else:
                    self._decode_str(phase=phase_name)
                self._fast_forward('"', phase="structure")

            def decode_object() -> None:
                """Decodes a nested JSON object."""
                self._fast_forward('{', phase="structure")
                self._decode_properties(param_schema.properties or {})
                self._fast_forward('}', phase="structure")

            def decode_num(allowed_chars: str) -> None:
                """Decodes a numeric parameter.

                Args:
                    allowed_chars (str): Characters allowed after the number.
                """
                self._decode_number(
                    allowed_chars, phase=phase_name,
                    autocomplete_options=get_candidates(param_schema.type)
                )

            decoders: dict[str, Any] = {
                "string": decode_string,
                "boolean": lambda: self._decode_bool(phase=phase_name),
                "bool": lambda: self._decode_bool(phase=phase_name),
                "number": lambda: decode_num(
                    "}" if i == len(keys) - 1 else ","
                ),
                "integer": lambda: decode_num(
                    "}" if i == len(keys) - 1 else ","
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
        """Bypasses LLM generation to append hardcoded structural tokens.

        Args:
            text (str): The exact structural string to append.
            phase (str): The current generation phase for the visualizer.
        """
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
        logit_boosts: Optional[dict[str, float]] = None,
        autocomplete_options: Optional[list[str]] = None
    ) -> str:
        """The core autoregressive generation loop.

        Args:
            get_valid_ids (Callable): Returns a list of valid token IDs.
            is_done (Callable): Returns (is_done, keep_latest_token).
            phase (str): Current phase for visualization (e.g. "param:a").
            logit_boosts (dict): Mapping from token substring to boost.
            autocomplete_options (list): Candidates to fast-forward.

        Returns:
            str: The generated and validated string segment.
        """
        current_value = ""
        token_count = 0

        # Precompute boosted token IDs if provided
        boost_tokens: dict[int, float] = {}
        if logit_boosts:
            boost_tokens = self.masker.get_boost_tokens(
                tuple(logit_boosts.items())
            )

        while True:
            logits = self.llm.get_logits(self._context_ids)
            valid_ids = get_valid_ids(current_value)

            masked_logits = np.full(len(logits), -np.inf)
            masked_logits[valid_ids] = np.array(logits)[valid_ids]

            # Apply logit boosting
            if logit_boosts and token_count > 3:
                # Flat boost initially, progressive scaling only for run-away
                # loops
                multiplier = 1.0 + max(0.0, float(token_count - 15)) * 0.5
                for vid, boost_val in boost_tokens.items():
                    if masked_logits[vid] > float("-inf"):
                        masked_logits[vid] += (boost_val * multiplier)

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

            if autocomplete_options:
                cv_strip = current_value.strip()
                matching_options = [
                    opt for opt in autocomplete_options
                    if opt.startswith(cv_strip)
                ]
                if len(matching_options) == 1:
                    suffix = matching_options[0][len(cv_strip):]
                    if suffix:
                        self._fast_forward(suffix, phase=phase)
                        current_value += suffix
                    break

            if done:
                break

        return current_value.strip()

    def _decode_enum(self, options: list[str], phase: str) -> str:
        """Decodes an enum string strictly matching one of the options.

        Args:
            options (list[str]): The allowed enum strings.
            phase (str): Current generation phase.

        Returns:
            str: The generated enum string.
        """
        return self._run_decode_loop(
            get_valid_ids=lambda val: self.masker.get_enum_tokens(
                options, val
            ),
            is_done=lambda val, latest_token: (
                (val + latest_token).strip() in options, True
            ),
            phase=phase,
            autocomplete_options=options
        )

    def _decode_str(
        self, phase: str, autocomplete_options: Optional[list[str]] = None
    ) -> str:
        """Decodes an arbitrary string up to the closing quote.

        Args:
            phase (str): Current generation phase.
            autocomplete_options (list[str]): Optional fast-forward strings.

        Returns:
            str: The generated string.
        """
        return self._run_decode_loop(
            get_valid_ids=lambda val: self.masker.get_string_tokens(),
            is_done=lambda val, latest_token: ('"' in latest_token, False),
            phase=phase,
            logit_boosts={'"': 5.0},
            autocomplete_options=autocomplete_options
        )

    def _decode_bool(self, phase: str) -> str:
        """Decodes a boolean 'true' or 'false' literal.

        Args:
            phase (str): Current generation phase.

        Returns:
            str: The generated boolean string.
        """
        return self._run_decode_loop(
            get_valid_ids=lambda val: self.masker.get_boolean_tokens(
                val
            ),
            is_done=lambda val, latest_token: (
                (val + latest_token).strip() in ["true", "false"], True
            ),
            phase=phase
        )

    def _decode_number(
        self, allowed_chars: str, phase: str,
        autocomplete_options: Optional[list[str]] = None
    ) -> str:
        """Decodes a numeric literal up to a stop character.

        Args:
            allowed_chars (str): Chars allowed after number.
            phase (str): Current generation phase.
            autocomplete_options (list[str]): Optional fast-forward strings.

        Returns:
            str: The generated numeric string.
        """
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
            logit_boosts=boosts,
            autocomplete_options=autocomplete_options
        )
