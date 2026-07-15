"""Module for test_tokenizer.py."""

import sys
import os
import pytest
from typing import Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("llm_sdk")
from llm_sdk import Small_LLM_Model  # type: ignore  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402


# Load model and tokenizers once for all tests
@pytest.fixture(scope="module")
def tokenizers() -> Tuple[BPETokenizer, Any]:
    """Executes tokenizers."""
    model = Small_LLM_Model()
    custom_tokenizer = BPETokenizer(
        model.get_path_to_vocab_file(),
        model.get_path_to_merges_file()
    )

    try:
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B",
            trust_remote_code=True
        )
    except ImportError:
        pytest.skip("transformers library not installed.")

    return custom_tokenizer, hf_tokenizer


TEST_STRINGS = [
    "Hello world! My name is GPT-2.",
    "Let's test some punctuation: , . : ; ' \" [ ] { }",
    "How about numbers? 12345 3.14159",
    "And some completely random string to ensure the numpy "
    "vectorization works perfectly!",
    "     Leading spaces and trailing spaces    ",
    "Testing\nnewlines\nand tabs",
]


@pytest.mark.parametrize("text", TEST_STRINGS)
def test_encode_matches_huggingface(
    tokenizers: Tuple[BPETokenizer, Any], text: str
) -> None:
    """Tests that encode matches huggingface."""
    custom_tokenizer, hf_tokenizer = tokenizers

    custom_ids = custom_tokenizer.encode(text)
    hf_ids = hf_tokenizer.encode(text)

    assert custom_ids == hf_ids, f"Encoded IDs do not match for '{text}'"


@pytest.mark.parametrize("text", TEST_STRINGS)
def test_decode_matches_huggingface(
    tokenizers: Tuple[BPETokenizer, Any], text: str
) -> None:
    """Tests that decode matches huggingface."""
    custom_tokenizer, hf_tokenizer = tokenizers

    # We use the HF encoder to get the ground truth IDs to decode
    hf_ids = hf_tokenizer.encode(text)

    custom_decoded = custom_tokenizer.decode(hf_ids)
    hf_decoded = hf_tokenizer.decode(hf_ids)

    assert custom_decoded == hf_decoded, \
        f"Decoded strings do not match for '{text}'"
    assert custom_decoded == text, \
        "Decoded string does not match original text"
