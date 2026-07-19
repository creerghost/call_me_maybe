import time
from typing import Optional
import numpy as np
from .models import GenerationEvent
from pydantic import ConfigDict, BaseModel, PrivateAttr


class Visualizer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id2token: dict[int, str]

    _start_time: Optional[float] = PrivateAttr(default=None)
    _last_time: Optional[float] = PrivateAttr(default=None)
    _token_count: int = PrivateAttr(default=0)

    def render(self, event: GenerationEvent) -> None:
        """Renders the current state of generation to the terminal.

        Args:
            event (GenerationEvent): The latest generation event
                containing all metrics.
        """
        now = time.time()
        if self._start_time is None:
            self._start_time = now
            self._last_time = now

        self._token_count += 1
        assert self._start_time is not None
        assert self._last_time is not None

        elapsed_total = now - self._start_time
        avg_tps = self._token_count / elapsed_total \
            if elapsed_total > 0 else 0.0

        elapsed_step = now - self._last_time
        inst_tps = 1.0 / elapsed_step if elapsed_step > 0 else 0.0
        self._last_time = now

        dashboard = "\033[2J\033[H"  # clear screen & cursor home
        dashboard += "\033[96m=== Constrained JSON Decoder ===\033[0m\n\n"
        dashboard += f"\033[93mUser Prompt:\033[0m {event.user_question}\n"
        dashboard += (
            f"\033[90mEncoded Prompt ({len(event.input_ids)} tokens):\033[0m "
        )
        dashboard += f"{str(event.input_ids[:10])}...\n\n"

        dashboard += (
            f"\033[92mCurrent Phase:\033[0m {event.current_phase} "
            f"(Source: {event.source})\n"
        )
        dashboard += (f"\033[92mSpeed:\033[0m {inst_tps:.1f} "
                      f"tokens/sec (Avg: {avg_tps:.1f} tokens/sec)\n\n")

        if event.fast_forwarded:
            dashboard += (
                "\033[94mAllowed Tokens:\033[0m 1 (Fast-Forwarding!)\n"
            )
        else:
            dashboard += (
                f"\033[94mAllowed Tokens:\033[0m {len(event.valid_ids)}\n"
            )

            if event.logits is not None:
                logits_np = np.array(event.logits)
                valid_logits = logits_np[event.valid_ids]
                valid_logits = valid_logits - np.max(valid_logits)
                exp_logits = np.exp(valid_logits)
                probs = exp_logits / np.sum(exp_logits)

                top_k = min(3, len(probs))
                top_indices_in_valid = np.argsort(probs)[-top_k:][::-1]

                dashboard += "\033[90mTop Alternatives:\033[0m "
                alts = []
                for idx in top_indices_in_valid:
                    tok_id = event.valid_ids[idx]
                    prob = probs[idx]
                    tok_str = self.id2token[tok_id].replace("Ġ", " ")
                    alts.append(f"'{tok_str}' ({prob * 100:.1f}%)")
                dashboard += " | ".join(alts) + "\n"

        dashboard += "\n\033[95mGenerated Token:\033[0m "
        dashboard += f"'{event.token_str}' (ID: {event.next_token_id})\n\n"

        colored_json = event.full_json_string.replace('"', '\033[36m"\033[0m')
        colored_json = colored_json.replace(":", "\033[33m:\033[0m")
        dashboard += f"{colored_json}\n"

        print(dashboard, end="", flush=True)
