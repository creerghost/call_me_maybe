from __future__ import annotations
from pydantic import BaseModel, model_validator, ConfigDict
from typing import Any, Literal, Self, Optional
from .fsm import JSONState


class FunctionParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "string",
        "number",
        "integer",
        "boolean",
        "bool",
        "object",
        "array",
        "enum",
    ]
    properties: dict[str, FunctionParameter] | None = None
    items: FunctionParameter | None = None


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionParameter

    @model_validator(mode="after")
    def check_empty_strings(self) -> Self:
        """Validates that no critical strings are empty or just whitespace.

        Returns:
            Self: The validated model instance.

        Raises:
            ValueError: If name, description, or parameter keys are empty.
        """
        if not self.name.strip():
            raise ValueError("Function name cannot be empty")
        if not self.description.strip():
            raise ValueError("Function description cannot be empty")
        for k in self.parameters.keys():
            if not k.strip():
                raise ValueError("Parameter name cannot be empty")
        return self


class TestPrompt(BaseModel):
    __test__ = False  # tells pytest to not treat it as a test
    model_config = ConfigDict(extra="forbid")
    prompt: str

    @model_validator(mode="after")
    def check_empty_prompt(self) -> Self:
        """Validates that the test prompt is not empty or just whitespace.

        Returns:
            Self: The validated model instance.

        Raises:
            ValueError: If the prompt string is empty.
        """
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        return self


class FunctionCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]


class SchemaNode(BaseModel):
    type: Literal[
        "object", "array", "string", "number", "integer", "boolean", "enum"
    ]
    options: list[str] | None = None  # for enums (like fn names)
    properties: dict[str, FunctionParameter] | None = None
    items: FunctionParameter | None = None
    remaining_keys: set[str] | None = None  # tracks which keys we havent seen

    def get_child_type(self, key: str | None = None) -> str:
        """Returns the type of the current value being parsed."""
        if self.type == "object":
            val_schema = None
            if self.properties and key:
                val_schema = self.properties.get(key)
                # nested schema properties use keys without quotes and
                # val_schema becomes None. -> decoder treats everything as
                # a string.
                if val_schema is None:
                    val_schema = self.properties.get(key.strip('"'))
        else:
            val_schema = self.items
        return val_schema.type if val_schema else "string"


class GenerationEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_question: str
    input_ids: list[int]
    state: JSONState
    old_state: JSONState
    fast_forwarded: bool
    valid_ids: list[int]
    token_str: str
    next_token_id: int
    full_json_string: str
    context: dict[str, Any]
    logits: Optional[list[float]] = None
