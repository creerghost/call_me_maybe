"""Entry point for the call_me_maybe application."""

import argparse
import json

from .catch import catch
from .decoder import ConstrainedDecoder
from .llm import LLM
from .loader import Loader
from .models import FunctionCallResult, TestPrompt
from .output import OutputWriter
from .prompt import PromptConstructor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition", help="Path to functions "
                        "definition json file")
    parser.add_argument("--input", help="Path to function calling json file")
    parser.add_argument("--output", help="Path to output file")
    parser.add_argument("--llm_path", help="Path to LLM")
    parser.add_argument("--llm_name", help="Name of LLM model (name of class)")
    parser.add_argument("--visual", action="store_true",
                        help="Enable live CLI dashboard rendering")
    parser.add_argument("--model", help="Path or name of the HF model to load")
    return parser


@catch
def load_loader(functions_definition: str, input_path: str) -> Loader:
    return Loader(functions_definition, input_path)


@catch
def load_llm(llm_path: str, llm_name: str, hf_model: str | None) -> LLM:
    return LLM(llm_path, llm_name, hf_model)


@catch
def build_prompt(loader: Loader, test_prompt: TestPrompt) -> str:
    return PromptConstructor.build_prompt(loader.fn_defs, test_prompt.prompt)


@catch
def generate_result(decoder: ConstrainedDecoder,
                    loader: Loader,
                    test_prompt: TestPrompt,
                    visualize: bool) -> FunctionCallResult:
    prompt = build_prompt(loader, test_prompt)
    if not visualize:
        print(f"User prompt: {test_prompt.model_dump().values()}")
        print("\n" + "=" * 60)
    generated_text = decoder.generate(prompt, test_prompt.prompt,
                                      loader.fn_defs, visualize)
    if not visualize:
        print(f"Output: {generated_text}\n")
    loads = json.loads(generated_text)

    return FunctionCallResult(
        prompt=test_prompt.prompt,
        name=loads["name"],
        parameters=loads["parameters"],
    )


@catch
def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full load → prompt → decode → write pipeline."""
    loader = load_loader(args.functions_definition, args.input)

    llm = load_llm(args.llm_path, args.llm_name, args.model)
    decoder = ConstrainedDecoder(llm)

    results = []
    for i, test_prompt in enumerate(loader.test_prompts):
        if not args.visual:
            print("=" * 60 + "\n")
            print(f"{i + 1}. prompt")
        res = generate_result(decoder, loader, test_prompt, args.visual)
        results.append(res)
        OutputWriter.write_output(results, args.output)


@catch
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
