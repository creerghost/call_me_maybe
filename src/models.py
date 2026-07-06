from pydantic import BaseModel, ValidationError, model_validator, ConfigDict
from typing import Any, Literal, Self
import pytest


class FunctionParameter(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal["string", "number"]


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionParameter

    @model_validator(mode='after')
    def check_empty_strings(self) -> Self:
        if not self.name.strip():
            raise ValueError("Function name cannot be empty")
        if not self.description.strip():
            raise ValueError("Function description cannot be empty")
        for k in self.parameters.keys():
            if not k.strip():
                raise ValueError("Parameter name cannot be empty")
        return self


class TestPrompt(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str

    @model_validator(mode='after')
    def check_empty_prompt(self) -> Self:
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        return self


class FunctionCallResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str
    name: str
    parameters: dict[str, Any]

    # === TESTS == FUNCTION PARAMETER ===


def test_function_parameter_valid() -> None:
    param = FunctionParameter(type="string")
    assert param.type == "string"


def test_function_parameter_invalid_wrong_text() -> None:
    with pytest.raises(ValidationError):
        invalid_type: Any = "boolean"
        FunctionParameter(type=invalid_type)


def test_function_parameter_invalid_empty() -> None:
    with pytest.raises(ValidationError):
        invalid_args: Any = {}
        FunctionParameter(**invalid_args)


def test_function_parameter_invalid_integer() -> None:
    with pytest.raises(ValidationError):
        invalid_type: Any = 1
        FunctionParameter(type=invalid_type)


def test_function_parameter_invalid_float() -> None:
    with pytest.raises(ValidationError):
        invalid_type: Any = 1.0
        FunctionParameter(type=invalid_type)

    # === TESTS == TEST PROMPT ===


def test_test_prompt_valid() -> None:
    tp = TestPrompt(prompt="What is the weather?")
    assert tp.prompt == "What is the weather?"


def test_test_prompt_invalid_empty() -> None:
    with pytest.raises(ValidationError):
        invalid_args: Any = {}
        TestPrompt(**invalid_args)


def test_test_prompt_invalid_wrong_key() -> None:
    with pytest.raises(ValidationError):
        invalid_args: Any = {"prompa": "hello"}
        TestPrompt(**invalid_args)


def test_test_prompt_invalid_empty_key_value() -> None:
    with pytest.raises(ValidationError):
        invalid_args: Any = {"": ""}
        TestPrompt(**invalid_args)


def test_test_prompt_invalid_whitespace() -> None:
    with pytest.raises(ValidationError):
        TestPrompt(prompt="   \n \t")

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
        invalid_args: Any = {}
        FunctionDefinition(**invalid_args)


def test_function_definition_invalid_wrong_key() -> None:
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
    param = FunctionParameter(type="number")
    with pytest.raises(ValidationError):
        FunctionDefinition(
            name="   \t",
            description="Adds numbers",
            parameters={"a": param},
            returns=param
        )


def test_function_definition_invalid_whitespace_parameter_key() -> None:
    param = FunctionParameter(type="number")
    with pytest.raises(ValidationError):
        FunctionDefinition(
            name="fn_add",
            description="Adds numbers",
            parameters={"   ": param},
            returns=param
        )


def test_function_definition_invalid_no_nested_object() -> None:
    with pytest.raises(ValidationError):
        invalid_returns: Any = "number"
        FunctionDefinition(
            name="fn_add",
            description="Adds numbers",
            parameters={},
            returns=invalid_returns
        )

    # === TESTS == FUNCTION CALL RESULT ===


def test_function_call_result_valid() -> None:
    res = FunctionCallResult(
        prompt="Add 2 and 3",
        name="fn_add",
        parameters={"a": 2, "b": 3}
    )
    assert res.prompt == "Add 2 and 3"
    assert res.name == "fn_add"
    assert res.parameters["a"] == 2


def test_function_call_result_invalid_missing() -> None:
    with pytest.raises(ValidationError):
        invalid_args: Any = {"prompt": "Hello"}
        FunctionCallResult(**invalid_args)


def test_function_definition_invalid_parameters_wrong_key() -> None:
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


if __name__ == "__main__":
    from pathlib import Path
    path_obj = Path(__file__)
    print(f"\n=== TESTING {path_obj.stem}.py ===\n")
    pytest.main(["-v", "-o", "python_classes=*Suite", __file__])
