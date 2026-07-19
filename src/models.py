from __future__ import annotations
from pydantic import BaseModel, model_validator, ConfigDict
from typing import Any, Literal, Self, Optional


class FunctionParameter(BaseModel):
    """Base configuration model."""

    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "string",
        "number",
        "integer",
        "boolean",
        "bool",
        "object",
        "enum",
    ]
    properties: dict[str, FunctionParameter] | None = None
    items: FunctionParameter | None = None
    options: list[str] | None = None


class FunctionDefinition(BaseModel):
    """Defines an available function and its parameter schema."""

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
    """Defines a user test prompt mapped to an expected function call."""

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
    """Represents the final, parsed output of the constrained generation."""

    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]


class GenerationEvent(BaseModel):
    """Event payload emitted after every single token is generated."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_question: str
    input_ids: list[int]
    source: Literal["hardcoded", "llm"]
    current_phase: str  # e.g. "name", "param:a", "structure"
    fast_forwarded: bool
    valid_ids: list[int]
    token_str: str
    next_token_id: int
    full_json_string: str
    logits: Optional[list[float]] = None
