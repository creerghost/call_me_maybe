# Advanced error recovery (Bonus part)

Production-grade constrained decoders often implement the advanced error recovery -- such as setting hard limits on maximum string lengths, or backtracking when the logit probability fo the closing token drops too low

## What to implement

### State-Bound Token Counters
e.g., max length constraints

**The problem**: Model can hallucinate if it forgets to close the string with an `"`.

**The solution**: Track how many tokens have been generated since entering the current state.

### Confidence Threshoding / Fallbacks

**The problem**: Sometimes model generates a valid token, but its logit value is extremely low. This means model is confused and likely going down a hallucination path.

**The solution**: Look at the logit score of the token before selecting it.

### Repetition Penalties

*SCRATCHED ENTIRELY* - not good approach for small LLMs

**The problem**: The model gets stuck repeating the same phrase over and over

**The solution**: Penalize tokens that have already been generated recently.

## Plan

1. Modify `ConstrainedDecoder` to track `state_token_count` and force-close long strings.

2. Add a basic logit penalty for recently generated tokens.

## Results

### Max Length Constraints and Repetition Penalties
```bash
[
  {
    "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "Programming is fun*vowels are cool! *",
      "regex": "v[aeiou]s*are.*",
      "replacement": "*vowels are cool! *"
    }
  },
  {
    "prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "The cat sat on the mat with another cat. The mouse was sitting on the mat too! {",
      "regex": "\\bcat(\\s+)?dog( \\w*\\b)(\\s+",
      "replacement": "The dog sat on the mat with another cat. The mouse was sitting on the mat too! {"
    }
  }
]
```
- It stopped generating (thanks to max length constraints)
- Because logit substracting was too high (-5.0), it was scared to chose english letters (`a`, `o`,...).
    - lower value to -0.5
    - add logit boost to the closing quote `"`
- Also, llama model is a chat model, it thought "*I should keep talking!*" and started rambling about different things.
    - I will remove repetition penalty entirely. Its good for larger models, but not in this case.

### Fixing things

```json
[
  {
    "prompt": "What is the sum of 265 and 345?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2,
      "b": 3
    }
  },
  {
    "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
    "name": "fn_get_square_root",
    "parameters": {
      "a": 3
    }
  },
  {
    "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "Programming is fun",
      "regex": "v[aeiou]s",
      "replacement": "*"
    }
  },
  {
    "prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "The cat sat on the mat with another cat",
      "regex": "cat",
      "replacement": "dog"
    }
  }
]
```

Here, the model chose perfect source strings, but was struggling with regex in case of vowels replacement.