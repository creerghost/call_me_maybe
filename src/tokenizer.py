"""Module for custom Byte-Pair Encoding (BPE)."""

import numpy as np
import json
import re

""" Example of merges.txt file
#version: 0.2
Ġ t
i n
h e
r e
Ġt h
e r
"""


class BPETokenizer:
    """Regex-based BPE tokenizer matching HuggingFace specs."""
    def __init__(self, vocab_file: str, merges_file: str):
        # vocab_file contains all tokens the model already have
        # merges_file contains an ordered list of priority pairs
        # bpe works by iteratively squashing two tokens into one
        # merges_file sets an order which 2 tokens squash first

        """Initializes the instance."""
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)
        self.vocab_rev: dict[int, str] = {v: k for k, v in self.vocab.items()}
        with open(merges_file, "r", encoding="utf-8") as f:
            bpe_data = f.read().split("\n")
            if bpe_data and bpe_data[0].startswith("#version"):
                bpe_data = bpe_data[1:]
            self.bpe_ranks: dict[tuple[str, str], int] = {}
            for rank, line in enumerate(bpe_data):
                parts = line.split()
                if len(parts) == 2:
                    # gives every pair a priority score
                    # lower rank = higher priority
                    self.bpe_ranks[(parts[0], parts[1])] = rank
        self.byte_encoder: dict[int, str] = self.bytes_to_unicode()
        self.byte_decoder: dict[str, int] = {
            v: k for k, v in self.byte_encoder.items()
        }
        self.cache: dict[str, str] = {}
        # self.pat splits raw text into a list of words, spaces and punctuation
        # ensures that BPE doesn't merge the end of one word with the beggining
        # of the next word
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""
        )

    @staticmethod
    def bytes_to_unicode() -> dict[int, str]:
        """
        LLMs process raw UTF-8 bytes (0 - 255 numbers), not text.
        Raw bytes contain invisible control characters which break JSON.
        Function maps all 256 raw bytes to visible characters.
        (this is why space byte convers into weird G)
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
        """Executes bpe."""
        if token in self.cache:
            return self.cache[token]

        word = np.array(list(token), dtype=object)

        while len(word) > 1:
            # get all adjacent pairs
            pairs = list(zip(word[:-1], word[1:]))
            # find the pair with the lowest rank (highest prior)
            best_pair = min(
                pairs, key=lambda p: self.bpe_ranks.get(p, float("inf"))
            )
            # break if no merges can be made
            if best_pair not in self.bpe_ranks:
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
        self.cache[token] = res
        return res

    def encode(self, text: str) -> list[int]:
        """Executes encode."""
        bpe_token_ids = []
        # split text into words via regex
        matches: list[str] = re.findall(self.pat, text)
        for token in matches:
            # convert to utf-8 bytes, then map to our unicode chars
            # " hello" -> " Ghello"
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )
            # apply bpe, returns space separated strings of tokens
            bpe_str: str = self.bpe(token)
            for bpe_token in bpe_str.split(" "):
                if bpe_token in self.vocab:
                    bpe_token_ids.append(self.vocab[bpe_token])
        return bpe_token_ids

    def decode(self, tokens: list[int]) -> str:
        # look up the string for each id and join them into one
        """Executes decode."""
        text = "".join([self.vocab_rev.get(token, "") for token in tokens])

        bytes: list[int] = [self.byte_decoder[c] for c in text]
        byte_arr = bytearray(bytes)
        return byte_arr.decode("utf-8", errors="replace")
