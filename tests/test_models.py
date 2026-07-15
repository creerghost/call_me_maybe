from pydantic import ValidationError
from typing import Any
from src.models import (
    FunctionParameter,
    FunctionDefinition,
    TestPrompt,
    FunctionCallResult,
    SchemaNode,
)
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


# --- Edge cases: Empty strings ---


def test_function_definition_invalid_empty_name() -> None:
    """Tests that empty function names raise a ValidationError."""
    param = FunctionParameter(type="string")
    with pytest.raises(ValidationError):
        FunctionDefinition(
            name="",
            description="desc",
            parameters={"a": param},
            returns=param,
        )


def test_function_definition_invalid_whitespace_description() -> None:
    """Tests that whitespace-only descriptions raise a ValidationError."""
    param = FunctionParameter(type="string")
    with pytest.raises(ValidationError):
        FunctionDefinition(
            name="fn_test",
            description="   \t\n  ",
            parameters={"a": param},
            returns=param,
        )


def test_test_prompt_invalid_empty_string() -> None:
    """Tests that an empty string prompt raises a ValidationError."""
    with pytest.raises(ValidationError):
        TestPrompt(prompt="")


def test_test_prompt_invalid_newline_only() -> None:
    """Tests that a newline-only prompt raises a ValidationError."""
    with pytest.raises(ValidationError):
        TestPrompt(prompt="\n\n\n")


def test_test_prompt_invalid_tab_only() -> None:
    """Tests that a tab-only prompt raises a ValidationError."""
    with pytest.raises(ValidationError):
        TestPrompt(prompt="\t\t\t")


# --- Edge cases: Large numbers ---


def test_function_call_result_large_integer() -> None:
    """Tests that very large integers are accepted in results."""
    res = FunctionCallResult(
        prompt="Add big numbers",
        name="fn_add",
        parameters={"a": 999999999999999, "b": 888888888888888},
    )
    assert res.parameters["a"] == 999999999999999


def test_function_call_result_negative_numbers() -> None:
    """Tests that negative numbers are accepted in results."""
    res = FunctionCallResult(
        prompt="Subtract",
        name="fn_sub",
        parameters={"a": -42, "b": -999},
    )
    assert res.parameters["a"] == -42


def test_function_call_result_float_precision() -> None:
    """Tests that high-precision floats are accepted in results."""
    res = FunctionCallResult(
        prompt="Pi calculation",
        name="fn_calc",
        parameters={"value": 3.141592653589793},
    )
    assert abs(res.parameters["value"] - 3.141592653589793) < 1e-15


def test_function_call_result_zero_values() -> None:
    """Tests that zero values are accepted in results."""
    res = FunctionCallResult(
        prompt="Zero test",
        name="fn_zero",
        parameters={"a": 0, "b": 0.0},
    )
    assert res.parameters["a"] == 0
    assert res.parameters["b"] == 0.0


# --- Edge cases: Special characters ---


def test_test_prompt_unicode() -> None:
    """Tests that unicode characters are accepted in prompts."""
    tp = TestPrompt(prompt="Какое расстояние от Москвы до Праги?")
    assert "Москвы" in tp.prompt


def test_test_prompt_emoji() -> None:
    """Tests that emoji characters are accepted in prompts."""
    tp = TestPrompt(prompt="Calculate 🎲 + 🎯")
    assert "🎲" in tp.prompt


def test_function_call_result_special_chars() -> None:
    """Tests that special characters are accepted in result parameters."""
    res = FunctionCallResult(
        prompt="Regex replace",
        name="fn_regex",
        parameters={"pattern": r"\d+", "replacement": "***"},
    )
    assert res.parameters["pattern"] == r"\d+"


def test_test_prompt_quotes() -> None:
    """Tests that various quote characters are accepted in prompts."""
    tp = TestPrompt(prompt="Reverse the string 'hello \"world\"'")
    assert "hello" in tp.prompt


def test_function_call_result_backslash() -> None:
    """Tests that backslash sequences are accepted in result parameters."""
    res = FunctionCallResult(
        prompt="test",
        name="fn_test",
        parameters={"path": "C:\\Users\\test\\file.txt"},
    )
    assert "\\" in res.parameters["path"]


# --- Edge cases: Wrong types ---


def test_function_parameter_invalid_list() -> None:
    """Tests that a list as parameter type raises a ValidationError."""
    with pytest.raises(ValidationError):
        invalid_type: Any = ["string"]
        FunctionParameter(type=invalid_type)


def test_function_parameter_invalid_dict() -> None:
    """Tests that a dict as parameter type raises a ValidationError."""
    with pytest.raises(ValidationError):
        invalid_type: Any = {"type": "string"}
        FunctionParameter(type=invalid_type)


def test_function_parameter_invalid_none() -> None:
    """Tests that None as parameter type raises a ValidationError."""
    with pytest.raises(ValidationError):
        invalid_type: Any = None
        FunctionParameter(type=invalid_type)


def test_function_parameter_invalid_bool() -> None:
    """Tests that a bool as parameter type raises a ValidationError."""
    with pytest.raises(ValidationError):
        invalid_type: Any = True
        FunctionParameter(type=invalid_type)


def test_function_definition_invalid_int_name() -> None:
    """Tests that a non-string function name raises a ValidationError."""
    param = FunctionParameter(type="string")
    with pytest.raises(ValidationError):
        invalid_name: Any = 42
        FunctionDefinition(
            name=invalid_name,
            description="test",
            parameters={"a": param},
            returns=param,
        )


def test_function_definition_invalid_list_params() -> None:
    """Tests that a list instead of dict for parameters raises a
    ValidationError."""
    param = FunctionParameter(type="string")
    with pytest.raises(ValidationError):
        invalid_params: Any = [param]
        FunctionDefinition(
            name="fn_test",
            description="test",
            parameters=invalid_params,
            returns=param,
        )


def test_function_call_result_invalid_int_name() -> None:
    """Tests that a non-string result name raises a ValidationError."""
    with pytest.raises(ValidationError):
        invalid_name: Any = 123
        FunctionCallResult(
            prompt="test",
            name=invalid_name,
            parameters={},
        )


# --- Edge cases: Ambiguous and extreme prompts ---


def test_test_prompt_very_long() -> None:
    """Tests that very long prompts are accepted without error."""
    long_prompt = "a " * 5000
    tp = TestPrompt(prompt=long_prompt.strip())
    assert len(tp.prompt) > 9000


def test_test_prompt_single_character() -> None:
    """Tests that a single character is accepted as a valid prompt."""
    tp = TestPrompt(prompt="?")
    assert tp.prompt == "?"


def test_test_prompt_numeric_only() -> None:
    """Tests that a purely numeric prompt is accepted."""
    tp = TestPrompt(prompt="42")
    assert tp.prompt == "42"


# --- Edge cases: Multiple and complex parameters ---


def test_function_definition_many_parameters() -> None:
    """Tests that function definitions with many parameters are accepted."""
    params = {
        f"param_{i}": FunctionParameter(type="string") for i in range(20)
    }
    fd = FunctionDefinition(
        name="fn_many_params",
        description="A function with many parameters",
        parameters=params,
        returns=FunctionParameter(type="string"),
    )
    assert len(fd.parameters) == 20


def test_function_definition_mixed_types() -> None:
    """Tests that mixed parameter types are accepted."""
    fd = FunctionDefinition(
        name="fn_mixed",
        description="Mixed type parameters",
        parameters={
            "name": FunctionParameter(type="string"),
            "age": FunctionParameter(type="number"),
            "active": FunctionParameter(type="boolean"),
            "count": FunctionParameter(type="integer"),
        },
        returns=FunctionParameter(type="string"),
    )
    assert fd.parameters["name"].type == "string"
    assert fd.parameters["age"].type == "number"
    assert fd.parameters["active"].type == "boolean"
    assert fd.parameters["count"].type == "integer"


def test_function_call_result_mixed_value_types() -> None:
    """Tests that results with mixed parameter value types are accepted."""
    res = FunctionCallResult(
        prompt="complex call",
        name="fn_complex",
        parameters={
            "name": "Alice",
            "age": 30,
            "score": 95.5,
            "active": True,
        },
    )
    assert isinstance(res.parameters["name"], str)
    assert isinstance(res.parameters["age"], int)
    assert isinstance(res.parameters["score"], float)
    assert isinstance(res.parameters["active"], bool)


def test_function_definition_nested_object() -> None:
    """Tests that nested object type parameters are accepted."""
    fd = FunctionDefinition(
        name="fn_nested",
        description="Nested params",
        parameters={
            "user": FunctionParameter(
                type="object",
                properties={
                    "name": FunctionParameter(type="string"),
                    "age": FunctionParameter(type="number"),
                },
            )
        },
        returns=FunctionParameter(type="boolean"),
    )
    assert fd.parameters["user"].type == "object"
    assert fd.parameters["user"].properties is not None
    assert "name" in fd.parameters["user"].properties


def test_function_definition_array_parameter() -> None:
    """Tests that array type parameters with items schema are accepted."""
    fd = FunctionDefinition(
        name="fn_sort",
        description="Sort numbers",
        parameters={
            "arr": FunctionParameter(
                type="array",
                items=FunctionParameter(type="number"),
            )
        },
        returns=FunctionParameter(
            type="array", items=FunctionParameter(type="number")
        ),
    )
    assert fd.parameters["arr"].type == "array"
    assert fd.parameters["arr"].items is not None
    assert fd.parameters["arr"].items.type == "number"


# --- Edge cases: SchemaNode ---


def test_schema_node_unknown_key() -> None:
    """Tests that unknown keys fall back to 'string' type."""
    node = SchemaNode(
        type="object",
        properties={"known": FunctionParameter(type="number")},
    )
    assert node.get_child_type("unknown") == "string"


def test_schema_node_none_key() -> None:
    """Tests that a None key falls back to 'string' type."""
    node = SchemaNode(
        type="object",
        properties={"a": FunctionParameter(type="number")},
    )
    assert node.get_child_type(None) == "string"


def test_schema_node_array_items() -> None:
    """Tests that array nodes return their item type."""
    node = SchemaNode(
        type="array",
        items=FunctionParameter(type="integer"),
    )
    assert node.get_child_type() == "integer"


def test_schema_node_quoted_key() -> None:
    """Tests that quoted keys are resolved correctly."""
    node = SchemaNode(
        type="object",
        properties={"age": FunctionParameter(type="number")},
    )
    assert node.get_child_type('"age"') == "number"
