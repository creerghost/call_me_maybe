# Resourses

https://huggingface.co/docs/transformers/tokenizer_summary
https://huggingface.co/docs/transformers/generation_strategies
https://www.aidancooper.co.uk/constrained-decoding/

## Example of successful constrained decoder
Here, HF-model `TinyLlama/TinyLlama-1.1B-Chat-v1.0` was used.
- Qwen is heavily fine-tuned on JSON structures. When it opens a `"` for a parameter, its internal probabilities strongly favor closing it with `"` as soon as the semantic value is complete. `TinyLlama` is a general chat model and isn't as strict.
- BUT, because our decoder strictly enforces the state machine, it *prevented* the model from breaking the JSON syntax (e.g., it coudln't suddenly write a newline or a random bracket).
    - It forced the model to stay in the `PARAM_VALUE` state.
    - But because the model's probability of generating closing quote `"` dropped to near zero, and we don't have a maximum token limit for values, it got trapped in an infinite loop of generating string characters.

```bash
=== Constrained JSON Decoder ===

User Prompt: Replace all vowels in 'Programming is fun' with asterisks

Current State: PARAM_VALUE (was PARAM_VALUE)
Allowed Tokens: 31853
Generated Token: 'ing'

{"name":"fn_substitute_string_with_regex","parameters":{"source_string":"Programming▁is▁fun*vowels*greeting*world!*greeting*world*greeting*world!*greeting*world!*greeting*world!*greeting*world!*greeting*world!*greeting*world!*greeting
^C
Keyboard interrupt. Bye!
```