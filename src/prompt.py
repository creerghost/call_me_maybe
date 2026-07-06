from pydantic import ValidationError
from .models import FunctionDefinition, FunctionParameter
from .catch import PromptConstructionError
import pytest


class PromptConstructor():
    @staticmethod
    def build_prompt(functions: list[FunctionDefinition],
                     user_prompt: str) -> str:
        if not functions:
            raise PromptConstructionError("No valid function definition(s)")
        if not user_prompt.strip():
            raise PromptConstructionError("User prompt is not defined")

        instruction_block = (
            "You are a function-calling assistant.\n"
            "Choose exactly one function from the catalog.\n"
            "Return only a JSON object with name and parameters.\n"
            "Do not include extra text."
        )

        func_lines = ["Available functions:"]
        for f in functions:
            func_lines.append(f"- Name: {f.name}")
            func_lines.append(f"  Description: {f.description}")
            func_lines.append("  Parameters:")
            for p_name, p in f.parameters.items():
                func_lines.append(f"   - {p_name}: {p.type}")
            func_lines.append(f"  Returns: {f.returns.type}")

        functions_block = "\n".join(func_lines)

        user_request_block = f"User request: \n{user_prompt.strip()}"

        return (
            f"{instruction_block}\n\n"
            f"{functions_block}\n\n"
            f"{user_request_block}"
        )

def test_build_prompt_valid() -> None:
    param = FunctionParameter(type="number")
    fd = FunctionDefinition(
        name="fn_add",
        description="Adds two numbers",
        parameters={"a": param},
        returns=param
    )
    prompt = "What is 5 + 5?"
    res = PromptConstructor.build_prompt([fd], prompt)
    assert "Available functions:" in res
    assert "Name: fn_add" in res
    assert f"User request: \n{prompt}" in res

def test_build_prompt_empty_functions() -> None:
    prompt = "What is 5 + 5?"
    with pytest.raises(PromptConstructionError):
        PromptConstructor.build_prompt([], prompt)

def test_build_prompt_empty_user_prompt() -> None:
    param = FunctionParameter(type="number")
    fd = FunctionDefinition(
        name="fn_add",
        description="Adds two numbers",
        parameters={"a": param},
        returns=param
    )
    with pytest.raises(PromptConstructionError):
        PromptConstructor.build_prompt([fd], "    ")

def test_build_prompt_wrong_functions() -> None:
    param = FunctionParameter(type="number")
    prompt = "What is 5 + 5?"
    fd1 = FunctionDefinition(
        name="fn_add",
        description="Adds two numbers",
        parameters={"a": param},
        returns=param
    )
    with pytest.raises(ValidationError):
        fd2 = FunctionDefinition(
        name="fn_add",
        description="Adds two numbers",
        parameters={"a": "string"},
        returns=param
    )
    # this line will never run tho
        PromptConstructor.build_prompt([fd1, fd2], prompt)


if __name__ == "__main__":
    from pathlib import Path
    path_obj = Path(__file__)
    print (f"\n=== TESTING {path_obj.stem}.py ===\n")
    pytest.main(["-v", "-o", "python_classes=*Suite", __file__])