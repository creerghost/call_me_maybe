UV = uv
PYTHON = $(UV) run python

FUNCTIONS_JSON = data/input/functions_definition.json
INPUT_JSON = data/input/function_calling_tests.json
OUTPUT_JSON = data/output/function_calling_results.json

install:
	$(UV) sync

run: install
	$(PYTHON) -m src.main \
		--functions_definition $(FUNCTIONS_JSON) \
		--input $(INPUT_JSON) \
		--output $(OUTPUT_JSON)

debug:
	$(PYTHON) -m pdb src/main.py

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