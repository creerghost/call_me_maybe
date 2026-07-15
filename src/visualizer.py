import time
import numpy as np
from .models import GenerationEvent

class Visualizer:
    def __init__(self, id2token: dict[int, str]):
        self.start_time = None
        self.token_count = 0
        self.id2token = id2token

    def render(self, event: GenerationEvent) -> None:
        if self.start_time is None:
            self.start_time = time.time()
        
        self.token_count += 1
        elapsed = time.time() - self.start_time
        tps = self.token_count / elapsed if elapsed > 0 else 0.0

        dashboard = "\033[2J\033[H"  # clear screen & cursor home
        dashboard += ("\033[96m=== Constrained JSON Decoder "
                      "===\033[0m\n\n")
        dashboard += f"\033[93mUser Prompt:\033[0m {event.user_question}\n"
        dashboard += f"\033[90mEncoded Prompt ({len(event.input_ids)} tokens):\033[0m {str(event.input_ids[:10])}...\n\n"
        
        path = "Root"
        if event.context and 'stack' in event.context:
            stack = event.context['stack']
            current_key = event.context.get('current_key')
            if len(stack) > 0:
                current_node = stack[-1]
                val_type = current_node.get_child_type(current_key)
                if current_key:
                    path = f"{current_key.strip('\"')} ({val_type})"
                else:
                    path = f"Object ({val_type})"
                    
        dashboard += (f"\033[92mCurrent State:\033[0m {event.state.name} "
                      f"(was {event.old_state.name})\n")
        dashboard += f"\033[92mContext Path:\033[0m {path}\n"
        dashboard += f"\033[92mSpeed:\033[0m {tps:.1f} tokens/sec\n\n"

        if event.fast_forwarded:
            dashboard += ("\033[94mAllowed Tokens:\033[0m 1 "
                          "(Fast-Forwarding!)\n")
        else:
            dashboard += (f"\033[94mAllowed Tokens:\033[0m "
                          f"{len(event.valid_ids)}\n")
            
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
                    alts.append(f"'{tok_str}' ({prob*100:.1f}%)")
                dashboard += " | ".join(alts) + "\n"

        dashboard += f"\n\033[95mGenerated Token:\033[0m '{event.token_str}' (ID: {event.next_token_id})\n\n"

        colored_json = event.full_json_string.replace('"', '\033[36m"\033[0m')
        colored_json = colored_json.replace(':', '\033[33m:\033[0m')
        dashboard += f"{colored_json}\n"

        print(dashboard, end="", flush=True)
