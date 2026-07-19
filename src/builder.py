from .models import GenerationEvent
from pydantic import BaseModel, PrivateAttr, ConfigDict
from .masker import ValueMasker
from .visualizer import Visualizer
from .llm import LLM


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
