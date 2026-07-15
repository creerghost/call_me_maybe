from pydantic import ValidationError
from typing import Any
from src.models import FunctionParameter, FunctionDefinition, TestPrompt, FunctionCallResult
import pytest

def test_function_parameter_valid() -> None:
    """Tests successful validation of a correct function parameter."""
    param = FunctionParameter(type="string")
    assert param.type == "string"


def test_function_parameter_invalid_wrong_text() -> None:
    """Tests that a ValidationError is raised for an invalid string type."""
    with pytest.raises(ValidationError):
        invalid_type: Any = "random_string"
        FunctionParameter(type=invalid_type)


def test_function_parameter_invalid_empty() -> None:
    """Tests that a ValidationError is raised for empty parameters."""
    with pytest.raises(ValidationError):
        invalid_args: Any = {}
        FunctionParameter(**invalid_args)


def test_function_parameter_invalid_integer() -> None:
    """Tests that a ValidationError is raised when type is an integer."""
    with pytest.raises(ValidationError):
        invalid_type: Any = 1
        FunctionParameter(type=invalid_type)


def test_function_parameter_invalid_float() -> None:
    """Tests that a ValidationError is raised when type is a float."""
    with pytest.raises(ValidationError):
        invalid_type: Any = 1.0
        FunctionParameter(type=invalid_type)


def test_test_prompt_valid() -> None:
    """Tests successful validation of a correct test prompt."""
    tp = TestPrompt(prompt="What is the weather?")
    assert tp.prompt == "What is the weather?"


def test_test_prompt_invalid_empty() -> None:
    """Tests that a ValidationError is raised for an empty prompt."""
    with pytest.raises(ValidationError):
        invalid_args: Any = {}
        TestPrompt(**invalid_args)


def test_test_prompt_invalid_wrong_key() -> None:
    """Tests that a ValidationError is raised when providing an invalid key."""
    with pytest.raises(ValidationError):
        invalid_args: Any = {"prompa": "hello"}
        TestPrompt(**invalid_args)


def test_test_prompt_invalid_empty_key_value() -> None:
    """Tests that a ValidationError is raised for empty keys and values."""
    with pytest.raises(ValidationError):
        invalid_args: Any = {"": ""}
        TestPrompt(**invalid_args)


def test_test_prompt_invalid_whitespace() -> None:
    """Tests that a ValidationError is raised for whitespace prompts."""
    with pytest.raises(ValidationError):
        TestPrompt(prompt="   \n \t")


def test_function_definition_valid() -> None:
    """Tests successful validation of a correct function definition."""
    param = FunctionParameter(type="number")

    fd = FunctionDefinition(
        name="fn_add",
        description="Adds two numbers",
        parameters={"a": param, "b": param},
        returns=param
    )
    assert fd.name == "fn_add"
    assert len(fd.parameters) == 2


def test_function_definition_invalid_empty() -> None:
    """Tests that a ValidationError is raised for an empty function
    definition."""
    with pytest.raises(ValidationError):
        invalid_args: Any = {}
        FunctionDefinition(**invalid_args)


def test_function_definition_invalid_wrong_key() -> None:
    """Tests that a ValidationError is raised when providing an incorrect
    key."""
    param = FunctionParameter(type="number")
    with pytest.raises(ValidationError):
        invalid_args: Any = {
            "nam": "fn_add",
            "description": "Adds numbers",
            "parameters": {},
            "returns": param
        }
        FunctionDefinition(**invalid_args)


def test_function_definition_invalid_extra_argument() -> None:
    """Tests that extra unconfigured arguments raise a ValidationError."""
    param = FunctionParameter(type="number")
    with pytest.raises(ValidationError):
        invalid_args: Any = {
            "name": "fn_add",
            "description": "Adds numbers",
            "parameters": {"a": param},
            "favorite_food": "pizza",
            "returns": param
        }
        FunctionDefinition(**invalid_args)


def test_function_definition_invalid_whitespace_name() -> None:
    """Tests that whitespace function names raise a ValidationError."""
    param = FunctionParameter(type="number")
    with pytest.raises(ValidationError):
        FunctionDefinition(
            name="   \t",
            description="Adds numbers",
            parameters={"a": param},
            returns=param
        )


def test_function_definition_invalid_whitespace_parameter_key() -> None:
    """Tests that whitespace parameter keys raise a ValidationError."""
    param = FunctionParameter(type="number")
    with pytest.raises(ValidationError):
        FunctionDefinition(
            name="fn_add",
            description="Adds numbers",
            parameters={"   ": param},
            returns=param
        )


def test_function_definition_invalid_no_nested_object() -> None:
    """Tests that a ValidationError is raised if a nested object is missing."""
    with pytest.raises(ValidationError):
        invalid_returns: Any = "number"
        FunctionDefinition(
            name="fn_add",
            description="Adds numbers",
            parameters={},
            returns=invalid_returns
        )


def test_function_call_result_valid() -> None:
    """Tests successful validation of a correct function call result."""
    res = FunctionCallResult(
        prompt="Add 2 and 3",
        name="fn_add",
        parameters={"a": 2, "b": 3}
    )
    assert res.prompt == "Add 2 and 3"
    assert res.name == "fn_add"
    assert res.parameters["a"] == 2


def test_function_call_result_invalid_missing() -> None:
    """Tests that a ValidationError is raised when required arguments are
    missing."""
    with pytest.raises(ValidationError):
        invalid_args: Any = {"prompt": "Hello"}
        FunctionCallResult(**invalid_args)


def test_function_definition_invalid_parameters_wrong_key() -> None:
    """Tests that a ValidationError is raised if a parameter key is invalid."""
    param = FunctionParameter(type="number")

    with pytest.raises(ValidationError):
        invalid_params: Any = {1: param, "b": param}
        invalid_returns: Any = "number"
        FunctionDefinition(
            name="fn_67",
            description="Raise 67 with your hands",
            parameters=invalid_params,
            returns=invalid_returns
        )


def test_function_definition_invalid_parameters_wrong_value() -> None:
    """Tests that a ValidationError is raised if a parameter value is
    invalid."""
    param = FunctionParameter(type="number")

    with pytest.raises(ValidationError):
        invalid_params: Any = {"a": param, "b": "broken"}
        invalid_returns: Any = "number"
        FunctionDefinition(
            name="fn_67",
            description="Raise 67 with your hands",
            parameters=invalid_params,
            returns=invalid_returns
        )
