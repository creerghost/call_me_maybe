from enum import Enum, auto
from typing import Any
from .models import SchemaNode


class JSONState(Enum):
    EXPECT_OBJECT_START = auto()
    EXPECT_ARRAY_START = auto()
    EXPECT_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    EXPECT_COMMA_OR_END = auto()
    DONE = auto()


class JSONStateMachine:
    def _transition_to_value(
            self, val_schema: Any, stack: list[SchemaNode]
    ) -> tuple[JSONState, str]:
        S = JSONState
        val_type = val_schema.type if val_schema else "string"

        if val_type == "object":
            new_node = SchemaNode(
                type="object",
                properties=val_schema.properties or {},
                remaining_keys=set(f'"{k}"'
                                   for k in (
                                    val_schema.properties
                                    or {}).keys())
            )
            stack.append(new_node)
            return S.EXPECT_OBJECT_START, ""
        elif val_type == "array":
            new_node = SchemaNode(
                type="array",
                items=val_schema.items
            )
            stack.append(new_node)
            return S.EXPECT_ARRAY_START, ""
        else:
            return S.EXPECT_VALUE, ""

    def transition_state(self, state: JSONState, current_prefix: str,
                         context: dict[str, Any]) -> tuple[JSONState, str]:
        # always looking at the top of the stack
        # tells us if we are currently inside an obj or an arr
        # and what keys we are still waiting for
        S = JSONState
        stack = context['stack']
        current_node: Any = stack[-1] if stack else None

        prefix_strip = current_prefix.strip()

        # expecting object/array? output { or [
        if state == S.EXPECT_OBJECT_START:
            if prefix_strip == "{":
                return S.EXPECT_KEY, ""

        elif state == S.EXPECT_ARRAY_START:
            if prefix_strip == "[":
                return self._transition_to_value(
                    current_node.items if current_node else None, stack)

        # checking if the token matches with remaining keys
        # if it does, save it and remove from remaining keys
        # llm can't generate it twice - genious
        elif state == S.EXPECT_KEY:
            # wait until the LLM generates a valid key from our remaining_keys
            if current_node and \
                    current_node.remaining_keys and \
                    prefix_strip in current_node.remaining_keys:
                context['current_key'] = prefix_strip
                current_node.remaining_keys.remove(prefix_strip)
                return S.EXPECT_COLON, ""

        # when we see a ':', we look up the schema for the key
        # we parsed
        elif state == S.EXPECT_COLON:
            if prefix_strip == ":":
                val_schema = current_node.properties.get(
                    context['current_key'])
                if val_schema is None:
                    val_schema = current_node.properties.get(
                        context['current_key'].strip('"'))

                return self._transition_to_value(val_schema, stack)

        elif state == S.EXPECT_VALUE:
            # figure out what type we are parsing based on the stack
            val_type = current_node.get_child_type(context.get('current_key'))

            if val_type in ("string", "enum"):
                if current_prefix.endswith('"') and len(prefix_strip) > 1:
                    # if we just parsed the function name, load its parameters
                    # into the root node
                    if context.get('current_key') == '"name"' \
                            and len(stack) == 1:
                        func_def = context['func_defs'].get(prefix_strip)
                        if func_def:
                            params_node = current_node.properties[
                                '"parameters"']
                            params_node.properties = func_def.parameters
                    return S.EXPECT_COMMA_OR_END, ""

            elif val_type in ("number", "integer"):
                if "}" in current_prefix or "]" in current_prefix:
                    stack.pop()
                    if not stack:
                        return S.DONE, ""
                    return S.EXPECT_COMMA_OR_END, ""
                elif "," in current_prefix:
                    return S.EXPECT_KEY \
                        if current_node.type == "object" \
                        else S.EXPECT_VALUE, ""

            elif val_type in ("boolean", "bool"):
                if prefix_strip in ("true", "false"):
                    return S.EXPECT_COMMA_OR_END, ""

        elif state == S.EXPECT_COMMA_OR_END:
            if prefix_strip == ",":
                if current_node.type == "object":
                    return S.EXPECT_KEY, ""
                else:
                    return self._transition_to_value(current_node.items, stack)
            elif prefix_strip in ("}", "]"):
                stack.pop()
                if not stack:
                    return S.DONE, ""
                return S.EXPECT_COMMA_OR_END, ""

        # Stay in the same state and keep accumulating
        # prefix if nothing matched
        return state, current_prefix
