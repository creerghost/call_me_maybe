# mypy: ignore-errors
import pytest

import numpy as np
from src.builder import JSONBuilder
from src.models import FunctionDefinition, FunctionParameter
from src.masker import ValueMasker

from src.llm import LLM


class DummyLLM(LLM):
    def __init__(self, **data):
        super().__init__(**data)
        self._token2id = {
            "fn_add": 0,
            "_numbers": 1,
            "2": 2,
            "3": 3,
            '"': 4,
            "true": 5,
            "false": 6,
            "hello": 7,
            " ": 8,
            ",": 9,
            "}": 10,
            "{": 11,
            ":": 12,
            "fn_add_numbers": 13,
        }
        self._id2token = {v: k for k, v in self._token2id.items()}
        self._token_strings = np.array(list(self._token2id.keys()))
        self._token_ids = np.array(list(self._token2id.values()))

    def model_post_init(self, __context):
        pass

    _mock_logits_fn = None

    def encode(self, prompt: str, **kwargs) -> list[int]:
        idx = hash(prompt)
        self._id2token[idx] = prompt
        return [idx]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id2token.get(i, "") for i in ids)

    def get_logits(self, ids: list[int]) -> np.ndarray:
        if self._mock_logits_fn:
            return self._mock_logits_fn(ids)
        return np.full(len(self._token2id), -np.inf)

    def get_vocab_size(self):
        return len(self._token2id)


@pytest.fixture
def mock_llm():
    return DummyLLM(llm_path="dummy", llm_name="dummy")


@pytest.fixture
def mock_masker(mock_llm):
    return ValueMasker(llm=mock_llm)


def test_builder_hardcoded(mock_llm, mock_masker):
    builder = JSONBuilder(llm=mock_llm, masker=mock_masker)
    builder._fast_forward('{"name": "', phase="test")
    assert builder._generated_text == '{"name": "'


def test_decode_enum(mock_llm, mock_masker):
    builder = JSONBuilder(llm=mock_llm, masker=mock_masker)
    builder._context_ids = []
    builder._generated_text = ""

    def get_logits_mock(ids):
        logits = np.full(len(mock_llm._token2id), -np.inf)
        if len(ids) == 0:
            logits[0] = 10.0  # "fn_add"
        else:
            logits[1] = 10.0  # "_numbers"
        return logits

    mock_llm._mock_logits_fn = get_logits_mock
    res = builder._decode_enum(options=["fn_add_numbers"], phase="name")

    assert res == "fn_add_numbers"
    assert builder._generated_text == "fn_add_numbers"


def test_decode_number(mock_llm, mock_masker):
    builder = JSONBuilder(llm=mock_llm, masker=mock_masker)
    builder._context_ids = []
    builder._generated_text = ""

    def get_logits_mock(ids):
        logits = np.full(len(mock_llm._token2id), -np.inf)
        if len(ids) == 0:
            logits[2] = 10.0  # "2"
        else:
            logits[10] = 10.0  # '}' (stop character)
        return logits

    mock_llm._mock_logits_fn = get_logits_mock
    builder._decode_number(allowed_chars="}", phase="param")

    assert builder._generated_text == "2"


def test_build_end_to_end(mock_llm, mock_masker):
    builder = JSONBuilder(llm=mock_llm, masker=mock_masker)

    fn_def = FunctionDefinition(
        name="fn_add_numbers",
        description="Add numbers",
        parameters={
            "a": FunctionParameter(type="number"),
            "b": FunctionParameter(type="number")
        },
        returns=FunctionParameter(type="number")
    )

    def get_logits_mock(ids):
        print(f"get_logits called, text: {builder._generated_text!r}")
        logits = np.full(len(mock_llm._token2id), -np.inf)
        if "fn_add_numbers" not in builder._generated_text:
            if "fn_add" not in builder._generated_text:
                logits[0] = 10.0  # fn_add
            else:
                logits[1] = 10.0  # _numbers
        elif ('"a":' in builder._generated_text and
              '"b":' not in builder._generated_text):
            if builder._generated_text.endswith('"a": '):
                logits[2] = 10.0  # "2"
            else:
                logits[9] = 10.0  # stop ','
        elif '"b":' in builder._generated_text:
            if builder._generated_text.endswith('"b": '):
                logits[3] = 10.0  # "3"
            else:
                logits[10] = 10.0  # stop '}'
        else:
            logits[4] = 10.0  # default to stop
        return logits

    mock_llm._mock_logits_fn = get_logits_mock

    result = builder.decode_function_call(
        fn_defs=[fn_def],
        prompt_ids=[99],
        user_question="sum 2 and 3"
    )

    assert result == (
        '{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}'
    )
