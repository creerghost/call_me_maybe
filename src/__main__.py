import argparse
import json

from .catch import catch
from .decoder import ConstrainedDecoder
from .llm import LLM
from .loader import Loader
from .models import FunctionCallResult, TestPrompt
from .output import OutputWriter
from .prompt import PromptConstructor
from .visualizer import Visualizer


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the CLI argument parser.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Run in interactive mode, prompting for user input"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to functions definition json file"
    )
    parser.add_argument(
        "--input",
        help="Path to function calling json file"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to output file"
    )
    parser.add_argument(
        "--llm_path", default="llm_sdk", help="Path to LLM"
    )
    parser.add_argument(
        "--llm_name", default="Small_LLM_Model",
        help="Name of LLM model (name of class)"
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Enable live CLI dashboard rendering",
    )
    parser.add_argument("--model", help="Path or name of the HF model to load")
    parser.add_argument(
        "--tokenizer",
        action="store_true",
        help="Use custom BPE tokenizer instead of HF",
    )
    return parser


def generate_result(
    decoder: ConstrainedDecoder,
    loader: Loader,
    test_prompt: TestPrompt,
    visualize: bool,
) -> FunctionCallResult:
    """Executes the constrained decoding process and returns the parsed result.

    Args:
        decoder (ConstrainedDecoder): The state-machine enforced decoder.
        loader (Loader): The JSON configuration loader.
        test_prompt (TestPrompt): The current user prompt object to process.
        visualize (bool): If True, renders the live CLI dashboard.

    Returns:
        FunctionCallResult: A parsed, validated Pydantic model of the output.
    """
    prompt = PromptConstructor.build_prompt(loader.fn_defs, test_prompt.prompt)

    visualizer = Visualizer(decoder.llm.id2token) if visualize else None

    generated_tokens = []

    for event in decoder.generate(
        prompt, test_prompt.prompt, loader.fn_defs
    ):
        if visualizer:
            visualizer.render(event)
        generated_tokens.append(event.next_token_id)

    generated_text = decoder.llm.decode(generated_tokens)
    loads = json.loads(generated_text)

    return FunctionCallResult(
        prompt=test_prompt.prompt,
        name=loads["name"],
        parameters=loads["parameters"],
    )


@catch
def run_pipeline(args: argparse.Namespace) -> None:
    """Runs the full load, prompt, decode, and output writing pipeline.

    Args:
        args (argparse.Namespace): The parsed command line arguments containing
            all file paths and execution flags.
    """
    loader = Loader(args.functions_definition, args.input)

    llm = LLM(
        args.llm_path, args.llm_name, args.model, tokenizer=args.tokenizer
    )
    decoder = ConstrainedDecoder(llm)

    if args.interactive:
        print("\n=== Interactive Mode ===")
        print("Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_input = input("\nEnter prompt: ")
                if user_input.strip().lower() in ("exit", "quit"):
                    break
                test_prompt = TestPrompt(prompt=user_input)
                res = generate_result(
                    decoder, loader, test_prompt, args.visual
                )
                print(f"\nResult: {res.model_dump_json(indent=2)}")
            except (EOFError, KeyboardInterrupt):
                print()
                break
    else:
        results = []
        for i, test_prompt in enumerate(loader.test_prompts):
            res = generate_result(decoder, loader, test_prompt, args.visual)
            results.append(res)
            OutputWriter.write_output(results, args.output)


@catch
def main() -> None:
    """Parses command line arguments and initiates the application pipeline."""
    parser = build_parser()
    args = parser.parse_args()
    if not args.interactive and not args.input:
        args.input = "data/input/function_calling_tests.json"
    run_pipeline(args)


if __name__ == "__main__":
    main()
