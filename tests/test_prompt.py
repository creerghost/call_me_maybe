from pydantic import ValidationError
from src.models import FunctionDefinition, FunctionParameter
from src.catch import PromptConstructionError
from src.prompt import PromptConstructor
from typing import Any
import pytest


def test_build_prompt_valid() -> None:
    """Tests successful construction of a prompt with valid inputs."""
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
    """Tests that an error is raised when the function list is empty."""
    prompt = "What is 5 + 5?"
    with pytest.raises(PromptConstructionError):
        PromptConstructor.build_prompt([], prompt)


def test_build_prompt_empty_user_prompt() -> None:
    """Tests that an error is raised when the user prompt is whitespace."""
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
    """Tests that invalid Pydantic schemas fail before prompt construction."""
    param = FunctionParameter(type="number")
    prompt = "What is 5 + 5?"
    fd1 = FunctionDefinition(
        name="fn_add",
        description="Adds two numbers",
        parameters={"a": param},
        returns=param
    )
    with pytest.raises(ValidationError):
        invalid_params: Any = {"a": "string"}
        fd2 = FunctionDefinition(
            name="fn_add",
            description="Adds two numbers",
            parameters=invalid_params,
            returns=param
        )
    # this line will never run tho
        PromptConstructor.build_prompt([fd1, fd2], prompt)


def test_build_prompt_multiple_functions() -> None:
    """Tests that prompts with multiple functions list all of them."""
    param_num = FunctionParameter(type="number")
    param_str = FunctionParameter(type="string")
    fn1 = FunctionDefinition(
        name="fn_add",
        description="Add numbers",
        parameters={"a": param_num, "b": param_num},
        returns=param_num,
    )
    fn2 = FunctionDefinition(
        name="fn_greet",
        description="Greet someone",
        parameters={"name": param_str},
        returns=param_str,
    )
    result = PromptConstructor.build_prompt(
        [fn1, fn2], "Add 5 and greet john"
    )
    assert "fn_add" in result
    assert "fn_greet" in result


def test_build_prompt_empty_string_prompt() -> None:
    """Tests that an empty string prompt raises PromptConstructionError."""
    param = FunctionParameter(type="number")
    fd = FunctionDefinition(
        name="fn_test",
        description="test",
        parameters={"a": param},
        returns=param,
    )
    with pytest.raises(PromptConstructionError):
        PromptConstructor.build_prompt([fd], "")
