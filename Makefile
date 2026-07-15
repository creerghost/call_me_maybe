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

RESET = \033[0m
BOLD = \033[1m
RED = \033[1;31m
GREEN = \033[1;32m
YELLOW = \033[1;33m
BLUE = \033[1;34m
MAGENTA = \033[1;35m
CYAN = \033[1;36m

all: help

install:
	@printf "$(CYAN)Syncing dependencies with uv...$(RESET)\n"
	$(UV) sync

run: install test
	@printf "$(MAGENTA)Running evaluator test suite...$(RESET)\n"
	$(PYTHON) -m src $(ARGS)

run-full: install
	@printf "$(MAGENTA)Running full suite (visual + tokenizer)...$(RESET)\n"
	$(PYTHON) -m src $(ARGS) --visual --tokenizer

run-interactive: install
	@printf "$(MAGENTA)Running interactive prompt mode...$(RESET)\n"
	$(PYTHON) -m src $(ARGS) --visual --interactive

run-visual: install
	@printf "$(MAGENTA)Running with visual mode...$(RESET)\n"
	$(PYTHON) -m src $(ARGS) --visual

run-custom: install
	@printf "$(MAGENTA)Running with custom model: $(MODEL_PATH)...$(RESET)\n"
	$(PYTHON) -m src $(ARGS) --model $(MODEL_PATH)

run-custom-visual: install
	@printf "$(MAGENTA)Running custom model with visual mode...$(RESET)\n"
	$(PYTHON) -m src $(ARGS) --model $(MODEL_PATH) --visual

run-custom-full: install
	@printf "$(MAGENTA)Running custom model (full)...$(RESET)\n"
	$(PYTHON) -m src $(ARGS) --model $(MODEL_PATH) --visual --tokenizer

test: install
	@printf "$(BLUE)Running tests...$(RESET)\n"
	$(PYTHON) -m pytest tests/ -v --no-header --tb=short
	@printf "$(GREEN)All tests passed!$(RESET)\n"


debug:
	@printf "$(YELLOW)Starting debugger...$(RESET)\n"
	$(PYTHON) -m pdb src/__main__.py

clean:
	@printf "$(RED)Cleaning all caches and environments...$(RESET)\n"
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .venv
	rm -rf data/output

clean-cache:
	@printf "$(RED)Cleaning caches...$(RESET)\n"
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	@printf "$(CYAN)Running standard linting...$(RESET)\n"
	$(UV) run flake8 src/ tests/
	$(UV) run mypy src/ tests/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@printf "$(CYAN)Running strict linting...$(RESET)\n"
	$(UV) run flake8 src/ tests/
	$(UV) run mypy src/ tests/ --strict

help:
	@printf "============================================================================\n"
	@printf "Welcome to the Makefile for the project!\n"
	@printf "This project has been created as a part of 42 curriculum by $(RED)vlnikola$(RESET).\n"
	@printf "\n"
	@printf "$(RED)First, ensure you have $(BOLD)$(GREEN)uv$(RED) installed and available in your environment.$(RESET)\n"
	@printf "\n"
	@printf "$(RED)Second, make sure to set paths to files in the data/input directory and the\n"
	@printf "LLM module path/name in the Makefile variables.$(RESET)\n"
	@printf "\n"
	@printf "Optionally, you can set the path to a custom model in the $(GREEN)MODEL_PATH$(RESET) variable.\n"
	@printf "\n"
	@printf "If you are not in 42 prague clusters, you should uncomment the following lines:\n"
	@printf "$(GREEN)  export HF_HOME = /sgoinfre/your_login/.cache/huggingface\n"
	@printf "  export TORCH_HOME = /sgoinfre/your_login/.cache/torch\n"
	@printf "\n"
	@printf "$(RESET)----------------------------------------------------------------------------$(RESET)\n"
	@printf "$(YELLOW)Available targets:$(RESET)\n"
	@printf "  $(GREEN)install$(RESET)            - Install dependencies using uv\n"
	@printf "  $(GREEN)run$(RESET)                - Run the comprehensive pytest suite for evaluators\n"
	@printf "  $(GREEN)run-visual$(RESET)         - Run the main script with visual mode\n"
	@printf "  $(GREEN)run-full$(RESET)           - Run the main script with visual mode and tokenizer\n"
	@printf "  $(GREEN)run-interactive$(RESET)    - Run the interactive prompt mode\n"
	@printf "  $(GREEN)run-custom$(RESET)         - Run the main script with custom model\n"
	@printf "  $(GREEN)run-custom-visual$(RESET)  - Run the main script with custom model and visual mode\n"
	@printf "  $(GREEN)run-custom-full$(RESET)    - Run the main script with custom model, visual mode, and tokenizer\n"
	@printf "  $(GREEN)test$(RESET)               - Run the test suite\n"
	@printf "  $(GREEN)debug$(RESET)              - Run the main script in debug mode using pdb\n"
	@printf "  $(GREEN)clean$(RESET)              - Remove temporary files and caches\n"
	@printf "  $(GREEN)clean-cache$(RESET)        - Remove only cache files\n"
	@printf "  $(GREEN)lint$(RESET)               - Run code linting and type checking\n"
	@printf "  $(GREEN)lint-strict$(RESET)        - Run strict code linting and type checking\n"
	@printf "  $(GREEN)help$(RESET)               - Show this help message\n"
	@printf "===========================================================================\n"

.PHONY: all install run run-full run-visual run-interactive run-custom run-custom-visual run-custom-full test debug clean clean-cache lint lint-strict help