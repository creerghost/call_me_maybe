from .models import GenerationEvent
from pydantic import BaseModel, PrivateAttr, ConfigDict
from .masker import ValueMasker
from .visualizer import Visualizer
from .llm import LLM
import numpy as np
from typing import Callable


class JSONBuilder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: LLM
    masker: ValueMasker
    visualizer: Visualizer | None = None

    _context_ids: list[int] = PrivateAttr(default_factory=list)
    _generated_text: str = PrivateAttr(default="")
    _user_question: str = PrivateAttr(default="")

    def _fast_forward(self, text: str, phase: str) -> None:
        # A case when we don't need llm
        token_ids = self.llm.encode(text)
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
        is_done: Callable[[str, str], bool],
        phase: str
    ) -> str:
        current_value = ""
        while True:
            logits = self.llm.get_logits(self._context_ids)
            valid_ids = get_valid_ids(current_value)

            masked_logits = np.full(len(logits), -np.inf)
            masked_logits[valid_ids] = np.array(logits)[valid_ids]
            next_token_id = int(np.argmax(masked_logits))

            token_str = self.llm.decode([next_token_id])
            self._context_ids.append(next_token_id)
            self._generated_text += token_str
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

            if is_done(current_value, token_str):
                break

        return current_value.strip()

    def _decode_enum(self, options: list[str], phase: str) -> str:
        return self._run_decode_loop(
            get_valid_ids=lambda val: self.masker.get_enum_tokens(
                options, val
            ),
            is_done=lambda val, latest_token: val.strip() in options,
            phase=phase
        )

    def _decode_str(self, phase: str) -> str:
        return self._run_decode_loop(
            get_valid_ids=lambda val: self.masker.get_string_tokens(),
            is_done=lambda val, latest_token: '"' in latest_token,
            phase=phase
        )
