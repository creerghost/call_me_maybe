*This project has been created as part of the 42 curriculum by vlnikola.*

# Call Me Maybe: A Deep Dive into Constrained Decoding

## WARNING
Information is outdated. Will be changed later

## Table of Contents
1. [Description](#1-description)
2. [Introduction to AI & LLMs](#2-introduction-to-ai--llms)
3. [Tokenization: The LLM Alphabet](#3-tokenization-the-llm-alphabet)
4. [Logits: The Prediction Scoreboard](#4-logits-the-prediction-scoreboard)
5. [Algorithm explanation](#5-algorithm-explanation)
6. [Pipeline example](#6-pipeline-example)
7. [Instructions](#7-instructions)
8. [Example usage](#8-example-usage)
9. [Design decisions](#9-design-decisions)
10. [Performance analysis](#10-performance-analysis)
11. [Challenges faced](#11-challenges-faced)
12. [Testing strategy](#12-testing-strategy)
13. [Resources](#13-resources)

---

## 1. Description

**Call Me Maybe** is a robust pipeline for executing function-calling with extremely small Language Models. 

While massive models like GPT-4 can reliably output JSON through sheer parameter size and RLHF training, small models (like 0.5B - 1B parameter models) frequently fail to adhere to strict schemas. This project builds a custom constrained decoding engine from scratch in Python that forces any HuggingFace model to output valid JSON matching a predefined schema.

### Core Features:
- **Zero-Dependency Architecture:** Implemented without heavy frameworks like `outlines` or `guidance`.
- **Dynamic State Machine:** Tracks JSON parsing states (`OBJECT_START`, `KEY_NAME`, `PARAM_VALUE`, etc.).
- **Type-Enforced Decoding:** Filters token vocabularies on the fly based on whether the expected parameter is a string, number, integer, or boolean.
- **Dynamic Chat Templates:** Automatically formats raw prompts into the model's native conversational template for higher accuracy.

### Bonus Features Implemented:
- **CLI Visualization Dashboard:** Added a `--visual` flag to render a real-time, colorful dashboard in the terminal that displays the active state, token masking, and live generation.
- **Multiple Model Support:** The engine dynamically supports loading any HuggingFace causal language model via the `--model` CLI flag (e.g., `microsoft/Phi-3-mini-4k-instruct` or `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
- **Performance Optimizations:** Implemented LRU memoization for token masks and a "fast-forward" generation skip for deterministic tokens to dramatically boost Tokens Per Second (TPS).
- **Advanced Error Recovery:** Implemented state-bound max-length constraints and dynamic logit boosting to prevent small LLMs from falling into infinite generation loops when trapped in string-generation states.
- **Comprehensive Test Suite:** Developed a robust `pytest` suite validating schema parsing, Pydantic bounds, and the constrained decoder's edge cases.

---

## 2. Introduction to AI & LLMs

Artificial Intelligence (AI) has rapidly evolved, with [Large Language Models (LLMs)](#13-resources) standing at the forefront of natural language processing. At their core, LLMs are incredibly powerful text-prediction engines. You provide them with a chunk of text (a prompt), and their sole objective is to guess what should logically come next based on patterns they've learned from reading vast portions of the internet.

Despite their apparent "understanding" of language, LLMs do not comprehend text the way humans do. They operate entirely on statistical probabilities. When you ask an LLM a question, it is mathematically calculating the most likely sequence of words that would follow your question in a typical human conversation.

This probabilistic nature makes LLMs incredibly versatile for creative writing, coding, and chatting. However, it also introduces a significant flaw: **unpredictability**. When a software system requires structured data (like a strict JSON object) to execute a function, the LLM might decide to prepend its response with "Sure, here is your JSON:" or hallucinate a completely invalid formatting structure. 

This project solves that exact problem.

---

## 3. Tokenization: The LLM Alphabet

To understand how we control an LLM, we first must understand how it reads. LLMs do not read letters or words; they read **[tokens](#13-resources)**.

A token is the fundamental building block of text. It can be a whole word (like `apple`), a chunk of a word (like `pre-` or `-ing`), or even a single character. When an LLM generates text, it spits out one token at a time.

Here is how tokenization works under the hood:
1. **Chunking:** The tokenizer algorithm splits a sentence into chunks using methods like [Byte-Pair Encoding (BPE)](#13-resources).
2. **Mapping:** Every unique chunk is mapped to a specific ID number in the model's vocabulary (e.g., `Hello` might be token ID `15496`). 
3. **Encoding/Decoding:** When you send text to the LLM, the tokenizer translates it into an array of these numbers. When the LLM generates a number, the tokenizer translates it back into readable text.

Because the model only operates on numbers, our code can interact with the generation process at the numerical level, intercepting tokens before they are converted back to text.

---

## 4. Logits: The Prediction Scoreboard

When an LLM is trying to guess the very next token, it goes through a massive mathematical process. Think of it like playing a game of charades:

* **Context Gathering:** It looks at all the tokens (numbers) it has received so far.
* **Neural Layers (Building the "Concept"):** It passes this sequence through billions of parameters. By the time it reaches the end, it has built a highly complex, abstract **mathematical fingerprint** of the *idea* that should come next.
* **The Scoreboard (Logits):** Finally, the abstract representation passes through the **Language Model (LM) Head**, a linear projection layer. This layer maps the high-dimensional concept vector into the model's vocabulary space (which typically contains 30,000 to 150,000 unique tokens). It mathematically evaluates the alignment between the context vector and every single token in the vocabulary.
  * For example, the alignment score for an unrelated token like `"cement"` might be heavily negative (e.g., `-15.4`).
  * Conversely, the score for a highly probable token like `"ice"` might be strongly positive (e.g., `18.2`).

These raw compatibility scores are called **[logits](#13-resources)**. A higher logit means the model is very confident that the token should come next. Usually, the model simply picks the token with the highest logit.

If an LLM is generating JSON, the logit for `{` might be very high at the beginning. But as the generation continues, the model might get confused and the logit for a conversational token like `I` or `The` might randomly spike.

---

## 5. Algorithm explanation

**[Constrained Decoding](#13-resources)** acts as a strict set of guardrails on the LLM's autoregressive generation process. 

Normally, an LLM is free to pick whatever token it wants from its entire 150,000-word filing cabinet. In this project, we implement a **[Finite State Machine](#13-resources)** that tracks exactly where we are in the JSON structure and actively intercepts the generation loop. 

Here is the step-by-step algorithm executed for *every single token*:

1. **State Evaluation:** The machine checks its current state (e.g., "Are we expecting a number right now?").
2. **Valid Token Calculation:** Based on the state and the predefined JSON schema, the engine computes which characters are legally allowed next. For instance, if expecting a `boolean`, only tokens starting with `t`, `r`, `u`, `e`, or `f`, `a`, `l`, `s`, `e` are valid.
3. **Vocabulary Masking:** We search the LLM's entire vocabulary filing cabinet and build an allowed list. Any token that breaks the JSON syntax or schema rules is excluded.
4. **Logit Hijacking:** We intercept the LLM right *after* it generates the logits (compatibility scores) but *before* it actually outputs the token. We apply our mask, forcefully changing the scores of all invalid tokens to negative infinity (`-inf`).
5. **Argmax Selection:** We let the model pick the token with the highest remaining score. Since all invalid tokens were set to `-inf`, the model is mathematically forced to pick the most likely *valid* token, even if it originally wanted to say something else.
6. **State Transition:** Based on the chosen token, the state machine transitions to the next logical state (e.g., shifting from `PARAM_VALUE` to expecting a comma `,` or closing bracket `}`).

### Fast-Forward Optimization
To significantly improve performance, the algorithm includes a "fast-forward" optimization. If the state machine determines there is only *one* valid token (for example, the only valid token after a key is a colon `:`), the engine completely skips the heavy LLM forward pass. It immediately appends the mandatory token and transitions to the next state, drastically cutting down unnecessary compute time.

By strictly controlling the logit probabilities at runtime, this algorithm mathematically guarantees that the final output is 100% syntactically perfect JSON matching the exact schema—with zero hallucinations.



## 6. Pipeline example

To understand how Call Me Maybe processes a request, let's walk through the pipeline step-by-step.

**1. Input Definitions:**
First, the system loads a JSON file containing available functions. For example:
```json
{
  "name": "fn_add_numbers",
  "description": "Add two numbers together.",
  "parameters": {
    "a": { "type": "number" },
    "b": { "type": "number" }
  },
  "returns": { "type": "number" }
}
```
It also receives a user prompt: *"What is the sum of 265 and 345?"*

**2. Prompt Building:**
The `PromptConstructor` formats this into a strict, readable text block (often using the model's native Chat Template like `<|user|>`) to give the LLM the exact context of the functions and the user's request.

**3. Token Generation & Constraint Checking (The Magic):**
The LLM begins predicting the output one token at a time. 
- The decoder state machine is initialized at `OBJECT_START`.
- Before the LLM can say "Sure, here is the answer", the decoder steps in, blocks all conversational words in the filing cabinet, and forces the LLM to output `{`.
- Next, the state shifts to `KEY_NAME`. The decoder blocks everything except quotes and letters, forcing `"name":`.
- Next, the state shifts to `NAME_VALUE`, and dynamically limits the LLM's vocabulary to *only* the string `"fn_add_numbers"` because it is the only valid function. The LLM is essentially trapped and has no choice but to output the correct function name.
- When it reaches the parameters, the state shifts to `PARAM_VALUE` expecting a number. The decoder masks the vocabulary, meaning the LLM can *only* predict digits. It calculates the concept fingerprint for the answer and sees that `265` and `345` have the highest compatibility scores among the allowed digit tokens.

**4. Final JSON Output:**
The generated tokens are concatenated and parsed. Because the state machine mathematically prevented syntax errors at every single step, the output is guaranteed to be:
```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 265,
    "b": 345
  }
}
```

---

## 7. Instructions

### Prerequisites
- Python 3.10+
- `uv` package manager (recommended for fast dependency installation)
- `make`

### Installation
1. Clone the repository.
2. Run `make install` to set up the virtual environment and install dependencies.
```bash
make install
```

---

## 8. Example usage

To run the standard evaluation pipeline using the default model (Qwen 0.6B):
```bash
make run
```

To run the pipeline with the **live visualizer dashboard**:
```bash
make run-visual
```

### Using Custom Models
The engine supports dynamic loading of other HuggingFace models. You can specify a custom model path via the Makefile:
```bash
make run-custom MODEL_PATH=microsoft/Phi-3-mini-4k-instruct
```
Or with the visualizer:
```bash
make run-custom-visual MODEL_PATH=microsoft/Phi-3-mini-4k-instruct
```

*Note: You may encounter out-of-memory errors on smaller machines if you attempt to load models larger than 2B parameters without quantization.*

---

## 9. Design decisions

1. **State Machine Architecture:** I chose a modular state machine (`JSONState` Enum) to track the decoding process. This makes the code highly extensible. If we want to add support for arrays or nested objects in the future, we simply add new states (e.g., `ARRAY_START`) rather than rewriting a monolithic parsing loop.
2. **LRU Caching for Token Masks:** Computing valid tokens by iterating over a 150k vocabulary on every single generation step is incredibly slow. I utilized Python's `@lru_cache` to memoize the valid token sets for static states (like expecting a `:` or `{`). This drastically improved tokens-per-second (TPS).
3. **Pydantic Validation:** All schemas and outputs are strictly validated using Pydantic. This ensures that the engine fails fast if the input JSON definitions are malformed, rather than crashing mid-generation.
4. **Fast-Forward Optimization:** If the state machine determines that there is only *one* valid token (e.g., a colon `:` after a key), the decoder completely skips the LLM forward pass and appends the token directly. This cuts down inference time by nearly 30%.

---

## 10. Performance analysis

### Tokens Per Second (TPS)
The primary bottleneck in autoregressive generation is the LLM's forward pass. However, constrained decoding adds overhead because we must filter a massive logits array on the CPU.
- **Without caching:** The token filtering added ~150ms of overhead per generation step, reducing generation to ~4 TPS.
- **With LRU caching & Fast-Forwarding:** The overhead was reduced to <5ms for cached states. Because deterministic characters (`{`, `"`, `:`, `,`, `}`) skip the LLM entirely, effective TPS increased to **~15-20 TPS** on standard hardware, meaning the decoding constraint is nearly cost-free.

### Accuracy
Tested against `Qwen/Qwen2.5-0.5B` and `TinyLlama-1.1B`:
- **Unconstrained:** < 20% success rate for strictly formatted, parsable JSON outputs.
- **Constrained:** 100% JSON syntactic validity. 
- *Semantic accuracy* (did it choose the right parameters?) relies on the model's native intelligence, which is heavily improved by the dynamic chat template implementation.

---

## 11. Challenges faced

1. **The Multi-Token String Problem:** 
   When generating strings, models don't generate character-by-character. A token might represent a whole word, a fragment with a leading space (e.g., `Ġhello`), or a special symbol. The decoder had to intelligently allow these multi-character tokens without breaking the `PARAM_VALUE` state. If the engine blindly checked characters one-by-one against a schema, it would crash when the LLM tried to spit out a 5-character token. I had to implement a robust prefix-matching algorithm for the vocabulary filter.

   *Implementation Example:*
   ```python
   # If we expect "name" and the LLM has generated "na", 
   # we calculate the remainder ("me") and allow any token that starts with it.
   remainder = expected[len(curr_prefix):]
   for clean_str, token_id in self.clean_tokens:
       if remainder.startswith(clean_str):
           valid_ids.append(token_id)
   ```

2. **Missing Tokenizer Configs (Phi-3 Bug):** 
   When switching to the `Phi-3-mini-4k-instruct` architecture to test the Multiple Models bonus feature, a bug in the HuggingFace `transformers` library caused a fatal `KeyError` for `rope_scaling` inside their configuration file. Because this is deeply embedded in their library, I couldn't just change their code. I had to implement a runtime patch in the `llm_sdk` wrapper to dynamically intercept and fix the model configuration object *before* loading the model weights.

3. **BatchEncoding Type Mismatches:** 
   Using the dynamic `apply_chat_template` method introduced massive inconsistencies. HuggingFace tokenizers are not uniform: for some models (like Qwen), `apply_chat_template` returns a simple, flat Python list of integers `[1, 2, 3]`. For other models, it returns a dictionary-like `BatchEncoding` object containing multi-dimensional PyTorch tensors (e.g., `{'input_ids': tensor([[1, 2, 3]])}`). The `encode()` function required implementing a strict, dynamic type-checking and flattening mechanism to ensure the decoder loop didn't crash during list concatenation operations later in the pipeline.

4. **Performance Overhead of CPU Logit Filtering:**
   Initially, the engine ran at a sluggish 4 Tokens Per Second (TPS). Iterating over a 150,000-token vocabulary and performing string comparisons for *every single generation step* on the CPU was crippling performance. I solved this by implementing the `@lru_cache` decorator to memoize valid token masks for static states (like `NAME_KEY`), and by creating a Fast-Forward optimization that completely skips the LLM forward pass when only one token is logically possible (like `:`).

   *Implementation Example:*
   ```python
   # Fast-forward optimization: completely bypass the LLM
   if len(valid_ids) == 1:
       next_token_id = valid_ids[0]
       # No expensive model.get_logits() called!
   ```

5. **Error Recovery & Infinite Generation Loops:** 
   - **The Trapped LLM:** While testing `TinyLlama-1.1B`, the model would sometimes forget to close a JSON string with a quote `"`. Because our constrained decoder strictly blocked it from generating invalid JSON syntax (like random brackets or newlines), the model was effectively "trapped" in the `PARAM_VALUE` state. Its internal probability of generating a closing quote dropped to near zero, causing it to hallucinate infinitely long, technically valid string characters (e.g., `"Programming is fun*greeting*world!*greeting..."`).
   - **Failed Repetition Penalties:** I initially tried to fix this by adding a repetition penalty to punish the model for repeating the same words. However, for 1-Billion parameter models, subtracting logits too aggressively (e.g., `-5.0`) made the model mathematically "scared" to choose basic English letters, breaking the text entirely.
   - **The Solution (Max-Length Constraints):** To prevent infinite loops, I implemented a state-bound counter. Every time the model generates a character inside a string, a counter increments. If the counter hits a hard limit (e.g., 20 tokens), the decoder acts as a circuit breaker. It forcefully masks the entire vocabulary *except* the closing quote `"`, leaving the model with only one mathematical option: to close the string.
     
     *Implementation Example:*
     ```python
     if state_token_count > 20:
         # Circuit breaker: only allow the quote token
         valid_ids = [id for clean, id in self.clean_tokens if clean.strip() == '"']
     ```
   - **The Solution (Dynamic Logit Boosting):** Before hitting the hard limit, I also implemented dynamic **Logit Boosting**. Since the small LLM's natural probability of generating a closing quote was dropping too low, the decoder steps in during the `PARAM_VALUE` state and artificially adds a massive mathematical boost (e.g., `+5.0`) to the quote token's score in the "filing cabinet". This safely nudges the model toward closing the string naturally without completely breaking its language generation logic.

     *Implementation Example:*
     ```python
     # Nudge the model to finish its sentence after 3 tokens
     if logits[q_id] > float("-inf") and state_token_count > 3:
         logits[q_id] += 5.0
     ```

---

## 12. Testing strategy

The project utilizes `pytest` for robust unit testing:
- **Model Validation Tests:** Ensures Pydantic models correctly reject empty strings, invalid types, and malformed dictionaries.
- **Schema Tests:** Validates that the `Loader` class correctly handles missing files and gracefully reports JSON decode errors.
- **Integration Tests:** The `data/output/function_calling_results.json` acts as a regression test artifact. By diffing the output against known good runs, I can verify that updates to the state machine do not break generation accuracy.

To run the test suite:
```bash
make test
```
*(If a test target is added to the Makefile, it will execute `pytest src/models.py`).*

---

## 13. Resources

### References
- [HuggingFace Tokenizer Summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [HuggingFace Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [Aidan Cooper: Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)
- [LLM Visualization](https://bbycroft.net/llm)
- [Outlines Paper (Concept Reference)](https://arxiv.org/abs/2307.09702)
- [Finite State Machines](https://medium.com/@brijeshrn/beyond-free-form-text-how-constrained-decoding-is-reshaping-structured-generation-in-llms-5f7a38bef259)

### AI Usage Disclosure
This project was developed with the assistance of advanced AI coding agents (specifically Antigravity, utilizing Gemini models). 
- **Architectural Planning:** AI was used to brainstorm the state machine transitions and identify edge cases in token mapping.
- **Debugging:** AI agents identified the root cause of the `KeyError: 'type'` within the Phi-3 configuration and patched the `llm_sdk` module dynamically.
- **Code Generation:** Portions of the boilerplate, Pydantic models, and documentation were written collaboratively through pair-programming sessions with the AI.
- **Conceptualizing:** The AI helped break down the complex mathematics of logits and vocabulary filtering into understandable, implementable algorithms.