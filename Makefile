# UNCOMMENT these lines if you are not in 42 prague clusters!
# The size of HDD space in 42 clusters is very limited.
LOGIN = vlnikola
# export HF_HOME = /sgoinfre/$(LOGIN)/.cache/huggingface
# export TORCH_HOME = /sgoinfre/$(LOGIN)/.cache/torch

# ========================= CHANGE THESE ======================================
FUNCTIONS_JSON = data/input/functions_definition.json
INPUT_JSON = data/input/function_calling_tests.json
OUTPUT_JSON = data/output/function_calling_results.json
LLM_PATH = llm_sdk
LLM_NAME = Small_LLM_Model
# Model can be:
MODEL_PATH = TinyLlama/TinyLlama-1.1B-Chat-v1.0
# MODEL_PATH = microsoft/Phi-3-mini-4k-instruct  # (requires more and more RAM)
# =============================================================================

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

run-full: install
	$(PYTHON) -m src $(ARGS) --visual --tokenizer

run-visual: install
	$(PYTHON) -m src $(ARGS) --visual

run-custom: install
	$(PYTHON) -m src $(ARGS) --model $(MODEL_PATH)

run-custom-visual: install
	$(PYTHON) -m src $(ARGS) --model $(MODEL_PATH) --visual

run-tests: install
	@echo "Running tests..."
	$(PYTHON) -m src.models
	$(PYTHON) -m src.loader
	$(PYTHON) -m src.prompt
	$(PYTHON) -m src.output
	@echo "All tests passed!"


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
	@echo "Welcome to the Makefile for the project!"
	@echo "This project has been created as a part of 42 curriculum by vlnikola."
	@echo ""
	@echo "First, ensure you have uvicorn installed and available in your environment."
	@echo ""
	@echo "Second, make sure to set paths to files in the data/input directory and the"
	@echo "LLM module path/name in the Makefile variables."
	@echo ""
	@echo "Optionally, you can set the path to a custom model in the MODEL_PATH variable."
	@echo ""
	@echo "If you are not in 42 prague clusters, you should uncomment the following lines:"
	@echo "  export HF_HOME = /sgoinfre/your_login/.cache/huggingface"
	@echo "  export TORCH_HOME = /sgoinfre/your_login/.cache/torch"
	@echo ""
	@echo "----------------------------------------------------------------------------"
	@echo "Available targets:"
	@echo "  install       - Install dependencies using uvicorn"
	@echo "  run           - Run the main script with specified arguments"
	@echo "  run-visual    - Run the main script with specified arguments and visual mode"
	@echo "  run-custom    - Run the main script with custom model"
	@echo "  run-custom-visual - Run the main script with custom model and visual mode"
	@echo "  run-tests     - Run tests"
	@echo "  debug         - Run the main script in debug mode using pdb"
	@echo "  clean         - Remove temporary files and caches"
	@echo "  clean-cache   - Remove only cache files"
	@echo "  lint          - Run code linting and type checking"
	@echo "  lint-strict   - Run strict code linting and type checking"
	@echo "  help          - Show this help message"
	@echo "==========================================================================="

.PHONY: all install run run-visual run-custom run-custom-visual run-tests debug clean clean-cache lint lint-strict help