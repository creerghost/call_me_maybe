# Constrained Decoding Optimizations

## 1. NumPy Vocabulary Vectorization

**Goal:** Eliminate Python `for` loops over the 150,000+ token vocabulary during FSM state transitions.

### Precomputing Arrays (`src/llm.py`)
Instead of keeping the vocabulary as a standard Python dictionary, we initialize parallel NumPy arrays at startup:
- `self.token_strings`: An array of `dtype=object` containing the cleaned strings (replacing `Ġ` with space).
- `self.token_ids`: A parallel array of `dtype=np.int32`.

### Vectorized Prefix Matching (`src/decoder.py`)
FSM states that enforce specific strings (like `"name"`) or generic strings now use NumPy's C-compiled character operations (`np.char`) instead of Python list comprehensions.
- **Prefix checking:** `np.char.startswith(token_strs, remainder)` checks all 150k strings in a single operation, returning a boolean mask.
- **Boolean Masking:** We index into the IDs array instantly using the mask: `self.llm.token_ids[mask]`.
- **Complex Logic:** For number validation, `np.vectorize(is_valid_num)` applies Python logic across the array.

## 2. PyTorch Logit Masking

**Goal:** Confine logit masking, boosting, and token selection to PyTorch tensor operations, replacing standard Python `list` iterations.

### Tensor Operations (`src/decoder.py`)
During generation, if more than one token is valid, we invoke the LLM and receive a raw list of logits. We now process this entirely in PyTorch:
- **Initialization:** `logits_t = torch.tensor(logits, dtype=torch.float32)`
- **Masking:** Instead of a `for` loop over all vocabulary indices to set `-inf`, we create a tensor filled with `-inf` and map only the `valid_ids_tensor` elements over: `mask[valid_ids_tensor] = logits_t[valid_ids_tensor]`.
- **Boosting:** `torch.where()` applies additive boosts (e.g., `+10.0` for commas or braces) directly to specific indices without any looping.
- **Selection:** `torch.argmax(mask).item()` extracts the winning token ID.

## 3. FSM Token Boundary Enforcement (The Spillover Bug)

**Problem:** The LLM was generating multi-character tokens like `", "` (Quote, comma, space, quote) while inside a string parameter. This caused the model to jump across FSM states, ruining the JSON syntax because it bypassed our forced FSM commas.

**Solution:** BPE tokenizers group characters statistically, not syntactically. To mathematically force the LLM to respect our FSM boundaries, we applied a strict mask to `_get_string_tokens()`:
- `no_quote_mask = np.char.find(token_strs, '"') == -1` bans **any** token containing a quote.
- `exact_quote_mask = np.char.strip(token_strs) == '"'` explicitly re-allows the token that is *exactly* just a quote.
- This prevents the LLM from outputting tokens that "spill over" into the next state, forcing it to close the string cleanly so our FSM can take over.
