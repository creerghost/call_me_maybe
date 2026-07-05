# COMMENT if you are not in 42 prague clusters!
export HF_HOME = /sgoinfre/vlnikola/.cache/huggingface
export TORCH_HOME = /sgoinfre/vlnikola/.cache/torch

# CHANGE THIS:
FUNCTIONS_JSON = data/input/functions_definition.json
INPUT_JSON = data/input/function_calling_tests.json
OUTPUT_JSON = data/output/function_calling_results.json
LLM_PATH = llm_sdk
LLM_NAME = Small_LLM_Model

# TODO: add a target to run tests with pytest

# TODO: add a target to run manually where we will ask
# user required arguments and then run the script with those arguments

MODEL_PATH = microsoft/Phi-3-mini-4k-instruct
UV = uv
PYTHON = $(UV) run python

ARGS = \
	--functions_definition $(FUNCTIONS_JSON) \
	--input $(INPUT_JSON) \
	--output $(OUTPUT_JSON) \
	--llm_path $(LLM_PATH) \
	--llm_name $(LLM_NAME)

all: help

install:
	$(UV) sync

run: install
	$(PYTHON) -m src $(ARGS)

run-visual: install
	$(PYTHON) -m src $(ARGS) --visual

run-custom: install
	$(PYTHON) -m src $(ARGS) --model $(MODEL_PATH)

run-custom-visual: install
	$(PYTHON) -m src $(ARGS) --model $(MODEL_PATH) --visual

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

clean-cache:
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	$(UV) run flake8 src/
	$(UV) run mypy -p src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 src/
	$(UV) run mypy -p src --strict

help:
	@echo "============================================================================"
	@echo "Welcome to the Makefile for the project!\n"
	@echo "First, ensure you have uvicorn installed and available in your environment.\n"
	@echo "Second, make sure to set paths to files in the data/input directory and the"
	@echo "LLM module path/name in the Makefile variables."
	@echo "----------------------------------------------------------------------------"
	@echo "Available targets:"
	@echo "  install       - Install dependencies using uvicorn"
	@echo "  run           - Run the main script with specified arguments"
	@echo "  debug         - Run the main script in debug mode using pdb"
	@echo "  clean         - Remove temporary files and caches"
	@echo "  clean-cache   - Remove only cache files"
	@echo "  lint          - Run code linting and type checking"
	@echo "  lint-strict   - Run strict code linting and type checking"
	@echo "  help          - Show this help message"
	@echo "==========================================================================="