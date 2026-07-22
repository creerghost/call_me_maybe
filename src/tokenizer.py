import numpy as np
import json
import re
from pydantic import BaseModel, ConfigDict, PrivateAttr

"""
Example of merges.txt file:
#version: 0.2
Ġ t
i n
h e
r e
Ġt h
e r
"""


from typing import Any


class BPETokenizer(BaseModel):
    """A custom Byte-Pair Encoding tokenizer implementation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vocab_file: str
    merges_file: str

    _vocab: dict[str, int] = PrivateAttr(default_factory=dict)
    _vocab_rev: dict[int, str] = PrivateAttr(default_factory=dict)
    _bpe_ranks: dict[tuple[str, str], int] = PrivateAttr(
        default_factory=dict
    )
    _byte_encoder: dict[int, str] = PrivateAttr(default_factory=dict)
    _byte_decoder: dict[str, int] = PrivateAttr(default_factory=dict)
    _cache: dict[str, str] = PrivateAttr(default_factory=dict)
    _pat: re.Pattern[str] = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""")

    @property
    def vocab(self) -> dict[str, int]:
        """Returns the dictionary mapping string tokens to integer IDs.

        Returns:
            dict[str, int]: The vocabulary mapping string tokens to IDs.
        """
        return self._vocab

    @property
    def vocab_rev(self) -> dict[int, str]:
        """Returns the dictionary mapping integer IDs to string tokens.

        Returns:
            dict[int, str]: Reverse vocabulary mapping IDs to tokens.
        """
        return self._vocab_rev

    def model_post_init(self, __context: Any) -> None:
        """Loads vocabulary and merge rules, and precomputes encoders.

        Args:
            __context (Any): Context for pydantic post-init.
        """
        # vocab_file contains all tokens the model already have
        # merges_file contains an ordered list of priority pairs
        # bpe works by iteratively squashing two tokens into one
        # merges_file sets an order which 2 tokens squash first
        with open(self.vocab_file, "r", encoding="utf-8") as f:
            self._vocab = json.load(f)
        self._vocab_rev = {v: k for k, v in self._vocab.items()}
        with open(self.merges_file, "r", encoding="utf-8") as f:
            bpe_data = f.read().split("\n")
            if bpe_data and bpe_data[0].startswith("#version"):
                bpe_data = bpe_data[1:]
            for rank, line in enumerate(bpe_data):
                parts = line.split()
                if len(parts) == 2:
                    # gives every pair a priority score
                    # lower rank = higher priority
                    self._bpe_ranks[(parts[0], parts[1])] = rank
        self._byte_encoder = self.bytes_to_unicode()
        self._byte_decoder = {
            v: k for k, v in self._byte_encoder.items()
        }
        # self._pat splits raw text into a list of words, spaces and
        # punctuation ensures that BPE doesn't merge the end of one word with
        # the beggining of the next word

    @staticmethod
    def bytes_to_unicode() -> dict[int, str]:
        """Maps 256 raw bytes to visible characters.

        LLMs process raw UTF-8 bytes (0 - 255 numbers), not text.
        Raw bytes contain invisible control characters which break JSON.
        This function maps all 256 raw bytes to visible characters.
        (this is why the space byte converts into the weird Ġ).

        Returns:
            dict[int, str]: Mapping from byte integer to unicode char.
        """
        bytes = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        chars = bytes[:]
        n = 0
        for b in range(2**8):
            if b not in bytes:
                bytes.append(b)
                chars.append(2**8 + n)
                n += 1
        chars_str = [chr(n) for n in chars]
        return dict(zip(bytes, chars_str))

    def bpe(self, token: str) -> str:
        # bpe merge loop avoiding recursion!!
        """Applies Byte-Pair Encoding merges to a single word token.

        Args:
            token (str): The unicode-mapped byte string of a word.

        Returns:
            str: Space-separated string of BPE subword tokens.
        """
        if token in self._cache:
            return self._cache[token]

        word = np.array(list(token), dtype=object)

        while len(word) > 1:
            # get all adjacent pairs
            pairs = list(zip(word[:-1], word[1:]))
            # find the pair with the lowest rank (highest prior)
            best_pair = min(
                pairs, key=lambda p: self._bpe_ranks.get(p, float("inf"))
            )
            # break if no merges can be made
            if best_pair not in self._bpe_ranks:
                break
            fst, snd = best_pair
            match_mask = (word[:-1] == fst) & (word[1:] == snd)
            # get the raw indeces where matches is True
            match_idxs = set(np.where(match_mask)[0])

            # rebuild the array, skipping over the merged tokens
            new_word = []
            i = 0
            while i < len(word):
                if i in match_idxs:
                    new_word.append(fst + snd)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = np.array(new_word, dtype=object)
        res = " ".join(word)
        self._cache[token] = res
        return res

    def encode(self, text: str) -> list[int]:
        """Encodes a full text string into a list of BPE token IDs.

        Args:
            text (str): The raw text string to encode.

        Returns:
            list[int]: A list of integer token IDs corresponding to the text.
        """
        bpe_token_ids = []
        # split text into words via regex
        matches: list[str] = re.findall(self._pat, text)
        for token in matches:
            # convert to utf-8 bytes, then map to our unicode chars
            # " hello" -> " Ghello"
            token = "".join(
                self._byte_encoder[b] for b in token.encode("utf-8")
            )
            # apply bpe, returns space separated strings of tokens
            bpe_str: str = self.bpe(token)
            for bpe_token in bpe_str.split(" "):
                if bpe_token in self.vocab:
                    bpe_token_ids.append(self.vocab[bpe_token])
        return bpe_token_ids

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of token IDs into a human-readable string.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.

        Returns:
            str: The decoded human-readable text string.
        """
        text = "".join([self._vocab_rev.get(token, "") for token in tokens])

        bytes: list[int] = [self._byte_decoder[c] for c in text]
        byte_arr = bytearray(bytes)

        # decode bytes back to string, ignoring errors for partial tokens
        return byte_arr.decode("utf-8", errors="replace")
