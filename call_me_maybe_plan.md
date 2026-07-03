# Call Me Maybe — Study Guide & Project Plan

> Covers **mandatory** + **bonus** parts from `call_me_maybe_subj.pdf`

---

## Part 1: What to Study (in order)

Study these topics **before** you start coding. Each builds on the previous one.

### 1. Tokenization & Vocabulary Files
> [!IMPORTANT]
> This is the foundation — everything else depends on understanding how text ↔ token IDs work.

- **What to learn:**
  - What BPE (Byte-Pair Encoding) and SentencePiece tokenization are
  - How a vocab file maps token IDs → string tokens (e.g., `{"42": "Ġhello"}`)
  - The `Ġ` prefix convention (represents a leading space)
  - Difference between `encode(text) → [int]` and `decode([int]) → text`
- **Resources:**
  - [HuggingFace Tokenizer Summary](https://huggingface.co/docs/transformers/tokenizer_summary)
  - [Karpathy's "Let's build GPT" tokenizer section](https://www.youtube.com/watch?v=kCc8FmEb1nY) (first 30 min)
  - Inspect the vocab file: load it as JSON, search for tokens like `{`, `"`, digits

### 2. LLM Inference Basics (Autoregressive Generation)
- **What to learn:**
  - How an LLM generates text **one token at a time**
  - The loop: `prompt → tokenize → model → logits → pick token → append → repeat`
  - What **logits** are (raw scores before softmax; higher = more likely)
  - Greedy decoding (pick argmax) vs. sampling
  - What an **EOS token** is and when to stop
- **Resources:**
  - [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  - [HuggingFace Generation Docs](https://huggingface.co/docs/transformers/generation_strategies)

### 3. Constrained Decoding ⭐
> [!IMPORTANT]
> This is the **core skill** the project tests. You won't pass without deeply understanding this.

- **What to learn:**
  - The idea: at each generation step, **mask invalid tokens** by setting their logits to `-inf`
  - How to track **parser state** (e.g., "I'm inside a JSON string" vs. "expecting a key")
  - How to determine which tokens are **valid** at each state
  - Using the vocab file to map token strings → IDs and filter logits accordingly
  - Why this guarantees 100% valid JSON even from a tiny 0.6B model
- **How it works step by step:**
  1. Model produces logits for all ~150k tokens
  2. You check current JSON parser state (expecting `{`, `"`, digit, `}`, etc.)
  3. You find all token IDs whose string would be valid at this position
  4. Set all other logits to `-inf`
  5. Pick the highest remaining logit → that's your next token
  6. Update parser state, repeat
- **Resources:**
  - [Outlines paper](https://arxiv.org/abs/2307.09702) — read the approach, but **do not use the library** (forbidden!)
  - [LMQL paper](https://arxiv.org/abs/2212.06094) — another constrained decoding approach
  - Think of it as building a **JSON state machine** that filters tokens

### 4. JSON Schema & Type Validation
- **What to learn:**
  - The structure of `functions_definition.json` (name, description, parameters with types, returns)
  - JSON types: `number`, `integer`, `string`, `boolean`
  - How to constrain the decoder to **only produce valid values** for each type:
    - `number` → digits, `.`, `-`, `e`
    - `integer` → digits, `-`
    - `string` → any characters between `""`
    - `boolean` → only `true` or `false`
  - How to constrain the `name` field to only produce one of the function names from the definitions

### 5. The `llm_sdk` API
- **What to learn** (only public methods allowed!):
  - `Small_LLM_Model` — the main class
  - `get_logits_from_input_ids(input_ids: List[int]) → List[float]` — core inference
  - `get_path_to_vocab_file() → str` — get the vocab JSON path
  - `encode(text: str) → Tensor` — text to token IDs
  - `decode(token_ids: List[int]) → str` — token IDs to text (optional)
- **Action item:** Copy `llm_sdk/` into your project root (next to `src/`), open the source, and read every public method

### 6. Pydantic Models
- **What to learn:**
  - Pydantic `BaseModel` for data validation
  - Defining models for: function definitions, test prompts, output results
  - Field validators, type coercion
- **Why:** The subject says "All classes must use pydantic for validation"

### 7. Python Project Structure & Tooling
- **What to learn:**
  - `python -m src` and `__main__.py` / `__init__.py`
  - `argparse` for CLI arguments
  - `flake8` + `mypy` flags and how to pass lint checks
  - PEP 257 docstrings (Google style)

---

## Part 2: Implementation Plan

### Architecture Overview

```mermaid
graph LR
    A["CLI (__main__.py)"] --> B["Load & Validate Inputs"]
    B --> C["For each prompt"]
    C --> D["Build LLM prompt"]
    D --> E["Constrained Decoder"]
    E --> F["Parse JSON output"]
    F --> G["Validate against schema"]
    G --> H["Write output JSON"]
    
    E --> |"token loop"| E
    
    subgraph "Constrained Decoder"
        E1["Get logits"] --> E2["Compute valid tokens"]
        E2 --> E3["Mask invalid → -inf"]
        E3 --> E4["Pick argmax"]
        E4 --> E5["Append token"]
    end
```

### Phase 0: Project Setup
- [x] Makefile with uv targets
- [x] `pyproject.toml` with dependencies
- [ ] Copy `llm_sdk/` into project root
- [ ] Create proper `src/__main__.py` with `argparse`
- [ ] Create `.gitignore` (exclude `.venv/`, `__pycache__/`, `data/output/`)
- [ ] Verify `uv sync` + `uv run python -m src --help` works

### Phase 1: Pydantic Models (`src/models.py`)
Define data models for everything:
```python
class FunctionParameter(BaseModel):
    type: str  # "number", "integer", "string", "boolean"

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionParameter

class TestPrompt(BaseModel):
    prompt: str

class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
```

### Phase 2: Input Loading & Validation (`src/loader.py`)
- Load `functions_definition.json` → `list[FunctionDefinition]`
- Load `function_calling_tests.json` → `list[TestPrompt]`
- Handle: missing files, invalid JSON, schema mismatches
- All errors → clear messages, no crashes

### Phase 3: LLM Wrapper (`src/llm.py`)
- Initialize `Small_LLM_Model`
- Load and parse the vocabulary file (`get_path_to_vocab_file()`)
- Build a **reverse vocab mapping**: `token_string → token_id`
- Helper methods:
  - `encode_prompt(text: str) → list[int]`
  - `get_next_logits(input_ids: list[int]) → list[float]`

### Phase 4: Constrained Decoder (`src/decoder.py`) ⭐ THE CORE
> [!IMPORTANT]
> This is where 80% of your effort goes. Get this right and the project works.

**JSON State Machine** — track where you are in the output JSON:
```
States:
  OBJECT_START       → expecting `{`
  KEY_NAME           → expecting `"name"` or `"parameters"`
  COLON              → expecting `:`
  VALUE_STRING       → expecting `"fn_xxx"`
  VALUE_NUMBER       → expecting digits/float
  VALUE_INTEGER      → expecting digits only
  VALUE_BOOLEAN      → expecting `true`/`false`
  OBJECT_END         → expecting `}` or `,`
  ...
```

**For each state**, compute valid token IDs:
1. Look at what characters/strings are valid at this position
2. Find all vocab tokens that match (prefix-match for multi-char tokens)
3. Build a mask, set invalid logits to `-inf`
4. Pick argmax from remaining logits

**Key implementation decisions:**
- The `name` field is constrained to only valid function names from the definitions
- Parameter keys are constrained to the exact keys from the function definition
- Parameter values are constrained by their type
- You need to handle multi-character tokens carefully (a token might be `": "` which spans a colon and value)

### Phase 5: Prompt Engineering (`src/prompt.py`)
- Build a prompt that tells the LLM what function to call
- Include function definitions in the prompt so the model has context
- Format: system prompt + function definitions + user prompt
- The prompt guides the model; constrained decoding guarantees the structure

### Phase 6: Main Pipeline (`src/__main__.py`)
```python
1. Parse CLI args (--functions_definition, --input, --output)
2. Load function definitions + test prompts
3. Initialize LLM + decoder
4. For each prompt:
   a. Build LLM prompt with function context
   b. Run constrained decoding → get JSON string
   c. Parse JSON → FunctionCallResult
   d. Append to results
5. Write results to output JSON file
```

### Phase 7: Output & Validation (`src/output.py`)
- Write `list[FunctionCallResult]` to JSON file
- Validate: all required keys present, types match definitions
- Create output directory if it doesn't exist

### Phase 8: Polish & Compliance
- [ ] All functions have type hints
- [ ] All functions/classes have PEP 257 docstrings
- [ ] `make lint` passes (flake8 + mypy)
- [ ] `make lint-strict` passes
- [ ] Graceful error handling everywhere
- [ ] Test with edge cases: empty strings, large numbers, special characters

### Phase 9: README.md
Required sections:
- [ ] Opening line: `*This project has been created as part of the 42 curriculum by vlnikola.*`
- [ ] Description
- [ ] Instructions (install, run, examples)
- [ ] Algorithm explanation (constrained decoding)
- [ ] Design decisions
- [ ] Performance analysis
- [ ] Challenges faced
- [ ] Testing strategy
- [ ] Resources + AI usage disclosure

---

### Phase 10: Bonus Features

| Bonus | Difficulty | What to do |
|-------|-----------|------------|
| Multiple LLM models | 🟢 Easy | Add a `--model` CLI flag, init different models |
| Recode tokenizer (encode/decode) | 🔴 Hard | Build your own `encode()` using vocab file + BPE merge rules, avoid calling `model.encode()` |
| Advanced error recovery | 🟡 Medium | Retry failed generations, fallback strategies, graceful degradation |
| Performance optimizations | 🟡 Medium | Cache logits, precompute valid token masks per state, batch processing |
| Comprehensive test suite | 🟢 Easy | pytest tests for models, loader, decoder, edge cases |
| Visualization of generation | 🟡 Medium | Print/log each token as it's generated, show masked tokens, probabilities |
| Nested function arguments | 🔴 Hard | Extend state machine to handle objects/arrays inside parameters |
| Public tokenizer encode/decode | 🔴 Hard | Implement BPE encode from scratch using vocab + merges file |
| Encoding/decoding + constrained decoding demo | 🟡 Medium | Show how your custom tokenizer feeds into the constrained decoder |

> [!TIP]
> **Recommended bonus order:** Test suite → Multiple models → Visualization → Error recovery → Performance opts → Tokenizer recode

---

## Deliverables Checklist

```
call_me_maybe/
├── src/
│   ├── __init__.py
│   ├── __main__.py      # CLI entry point
│   ├── models.py         # Pydantic models
│   ├── loader.py          # Input file loading
│   ├── llm.py             # LLM wrapper + vocab
│   ├── decoder.py         # Constrained decoder ⭐
│   ├── prompt.py          # Prompt building
│   └── output.py          # Output writing
├── llm_sdk/               # Copied from provided package
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
├── pyproject.toml
├── uv.lock
├── Makefile
├── .gitignore
└── README.md
```

> [!CAUTION]
> **Do NOT commit** `data/output/` — it's generated during peer review.
> **Do NOT use** pytorch, transformers, outlines, dspy, or any HuggingFace package.
> **Do NOT use** private methods/attributes from `llm_sdk`.
