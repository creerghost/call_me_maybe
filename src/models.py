from typing import Type
from pydantic import BaseModel, ValidationError
from typing import Any
import pytest


class FunctionParameter(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionParameter


class TestPrompt(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]

    # === TESTS == FUNCTION PARAMETER ===

def test_function_parameter_valid() -> None:
    param = FunctionParameter(type="some string")
    assert param.type == "some string"

def test_function_parameter_valid_big_string() -> None:
    string = "A" * 100000
    param = FunctionParameter(type=string)
    assert param.type == string

def test_function_parameter_invalid_empty() -> None:
    with pytest.raises(ValidationError):
        FunctionParameter()

def test_function_parameter_invalid_integer() -> None:
    with pytest.raises(ValidationError):
        FunctionParameter(type=1)

def test_function_parameter_invalid_float() -> None:
    with pytest.raises(ValidationError):
        FunctionParameter(type=1.0)

    # === TESTS == TEST PROMPT ===

def test_test_prompt_valid() -> None:
    tp = TestPrompt(prompt="What is the weather?")
    assert tp.prompt == "What is the weather?"

def test_test_prompt_invalid_empty() -> None:
    with pytest.raises(ValidationError):
        TestPrompt()

    # === TESTS == FUNCTION DEFINITION ===

def test_function_definition_valid() -> None:
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
    with pytest.raises(ValidationError):
        FunctionDefinition()

def test_function_definition_invalid_no_nested_object() -> None:
    with pytest.raises(ValidationError):
        FunctionDefinition(
            name="fn_add",
            description="Adds numbers",
            parameters={},
            returns="number"
        )

def test_function_definition_invalid_parameters_wrong_key() -> None:
    param = FunctionParameter(type="number")

    with pytest.raises(ValidationError):
        FunctionDefinition(
            name="fn_67",
            description="Raise 67 with your hands",
            parameters={1: param, "b": param},
            returns="number"
        )

def test_function_definition_invalid_parameters_wrong_value() -> None:
    param = FunctionParameter(type="number")

    with pytest.raises(ValidationError):
        FunctionDefinition(
            name="fn_67",
            description="Raise 67 with your hands",
            parameters={"left_hand": param, "right_hand": "broken"},
            returns="number"
        )

if __name__ == "__main__":
    from pathlib import Path
    path_obj = Path(__file__)
    print (f"\n=== TESTING {path_obj.stem}.py ===\n")
    pytest.main(["-v", "-o", "python_classes=*Suite", __file__])