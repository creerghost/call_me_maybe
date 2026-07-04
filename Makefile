# COMMENT if you are not in 42 prague clusters!
# export HF_HOME = /sgoinfre/vlnikola/.cache/huggingface
# export TORCH_HOME = /sgoinfre/vlnikola/.cache/torch

UV = uv
PYTHON = $(UV) run python

FUNCTIONS_JSON = data/input/functions_definition.json
INPUT_JSON = data/input/function_calling_tests.json
OUTPUT_JSON = data/output/function_calling_results.json
LLM_PATH = llm_sdk.llm_sdk
LLM_NAME = Small_LLM_Model


ARGS = \
	--functions_definition $(FUNCTIONS_JSON) \
	--input $(INPUT_JSON) \
	--output $(OUTPUT_JSON) \
	--llm_path $(LLM_PATH) \
	--llm_name $(LLM_NAME)

install:
	$(UV) sync

run: install
	$(PYTHON) -m src $(ARGS)

debug:
	$(PYTHON) -m pdb src/__main__.py

clean:
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .venv
	rm -rf data/output

lint:
	$(UV) run flake8 src/
	$(UV) run mypy -p src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 src/
	$(UV) run mypy -p src --strict