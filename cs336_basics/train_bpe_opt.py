from collections import defaultdict
import os
import regex as re

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = b"'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\\sa-zA-Z0-9]+|\\s+(?!\\S)|\\s+"


class BPETokenizerTrainer:
    def __init__(
        self,
    ):
        self.vocab = defaultdict(bytes)
        self.merges = []
        self.special_tokens = set()

    def train(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], **kwargs):
        self.initialize_special_tokens(special_tokens=special_tokens)
        self.initialize_vocab()
        pre_token = self.pre_tokenize(input_path=input_path, pattern="d")
        num_merges = vocab_size - len(self.vocab)
        curr_merge = 0

        pairs = defaultdict(int)
        for key, value in pre_token.items():
            for i in range(len(key) - 1):
                pairs[key[i : i + 2]] += value

        while curr_merge < num_merges:
            curr_merge += 1
            bytes_tuple_to_merge = self.find_max(pairs)[0]
            self.merges.append(bytes_tuple_to_merge)
            bytes_to_merge = self.bytes_tuple_to_bytes(bytes_tuple_to_merge)
            self.vocab[len(self.vocab)] = bytes_to_merge
            modified_pre_token = defaultdict(int)
            original_pre_token = defaultdict(int)
            for key, value in pre_token.items():
                new_key = []
                i = 0
                has_change = False
                while i < len(key):
                    if i + 1 < len(key) and key[i : i + 2] == bytes_tuple_to_merge:
                        new_key.append(bytes_to_merge)
                        i += 1
                        has_change = True
                    else:
                        new_key.append(key[i])
                    i += 1

                if has_change:
                    new_key_tuple = tuple(new_key)
                    modified_pre_token[new_key_tuple] = value
                    original_pre_token[key] = value

            for key, value in original_pre_token.items():
                del pre_token[key]
                for i in range(len(key) - 1):
                    pairs[key[i : i + 2]] -= value
                    if pairs[key[i : i + 2]] == 0:
                        del pairs[key[i : i + 2]]

            for key, value in modified_pre_token.items():
                pre_token[key] = value
                for i in range(len(key) - 1):
                    pairs[key[i : i + 2]] += value

        return (self.vocab, self.merges)

    def initialize_special_tokens(
        self,
        special_tokens: list[str] | None = None,
    ):
        self.special_tokens = set()
        if special_tokens is None:
            return
        for s in special_tokens:
            self.special_tokens.add(s.encode("utf-8", errors="ignore"))

    def pre_tokenize(self, input_path, pattern="ws"):
        pre_token = defaultdict(int)
        with open(input_path, "rb") as f:
            lines = f.readlines()
            for line in lines:
                if pattern == "ws":
                    tokens = line.replace(b"\n", b"").split(b" ")
                    for token in tokens:
                        pre_token[self.bytes_to_bytes_tuple(token)] += 1
                else:
                    for m in re.finditer(PAT, line):
                        pre_token[self.bytes_to_bytes_tuple(m.group(0))] += 1
        return pre_token

    def initialize_vocab(self):
        for i, t in enumerate(self.special_tokens):
            self.vocab[i] = t

        num_special_tokens = len(self.special_tokens)
        for i in range(256):
            self.vocab[i + num_special_tokens] = bytes([i])

    def bytes_to_bytes_tuple(self, input_bytes):
        bytes_list = []
        for i in input_bytes:
            bytes_list.append(bytes([i]))

        return tuple(bytes_list)

    def bytes_tuple_to_bytes(self, input_bytes_tuple):
        int_list = []
        for i in input_bytes_tuple:
            int_list.extend(list(i))
        return bytes(int_list)

    def find_max(self, pairs):
        max_item = ((), 0)
        for key, value in pairs.items():
            if value > max_item[1] or (value == max_item[1] and key > max_item[0]):
                max_item = (key, value)
        return max_item
