# Advanced error recovery (Bonus part)

- Production-grade constrained decoders often implement the advanced error recovery -- such as setting hard limits on maximum string lengths, or backtracking when the logit probability fo the closing token drops too low

## What to implement

### State-Bound Token Counters
e.g., max length constraints

**The problem**: Model can hallucinate if it forgets to close the string with an `"`.

**The solution**: Track how many tokens have been generated since entering the current state.

### Confidence Threshoding / Fallbacks

**The problem**: Sometimes model generates a valid token, but its logit value is extremely low. This means model is confused and likely going down a hallucination path.

**The solution**: Look at the logit score of the token before selecting it.

### Backtracking

**The problem**: The model makes a valid choice at Token A, but that choice leads to a dead end at Token B where `valid_tokens` becomes completely empty (0 valid).

**The solution**: Rewind the generation by one step and pick the second best option for Token A.

### Repetition Penalties

**The problem**: The model gets stuck repeating the same phrase over and over

**The solution**: Penalize tokens that have already been generated recently.

## Plan

1. Modify `ConstrainedDecoder` to track `state_token_count` and force-close long strings.

2. Add a basic logit penalty for recently generated tokens.

3. Implement a history stack to rewind generation when hitting a dead end.
