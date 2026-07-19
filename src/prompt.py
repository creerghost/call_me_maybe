from .models import FunctionDefinition
from .catch import PromptConstructionError
from pydantic import BaseModel


class PromptConstructor(BaseModel):
    """Constructs dynamic templates for LLM instructions."""

    @staticmethod
    def build_prompt(
        functions: list[FunctionDefinition], user_prompt: str
    ) -> str:
        """Constructs a system prompt instructing the LLM to output JSON.

        Args:
            functions (list[FunctionDefinition]): A list of available function
            schemas.
            user_prompt (str): The raw request from the user.

        Returns:
            str: The fully assembled prompt string.

        Raises:
            PromptConstructionError: If the function list is empty or the user
            prompt is missing.
        """
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
            f"{instruction_block}\n\n{functions_block}\n\n{user_request_block}"
        )
