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

        pairs = defaultdict(list)
        for idx, item in enumerate(pre_token):
            key, value = item
            for i in range(len(key) - 1):
                if key[i : i + 2] not in pairs:
                    pairs[key[i : i + 2]] = [0, set()]
                pairs[key[i : i + 2]][0] += value
                pairs[key[i : i + 2]][1].add(idx)

        while curr_merge < num_merges:
            curr_merge += 1
            max_item = self.find_max(pairs)
            bytes_tuple_to_merge = max_item[0]
            token_idx_list = max_item[1][1]
            self.merges.append(bytes_tuple_to_merge)
            bytes_to_merge = self.bytes_tuple_to_bytes(bytes_tuple_to_merge)
            self.vocab[len(self.vocab)] = bytes_to_merge
            modified_pre_token = defaultdict(tuple)
            original_pre_token = defaultdict(tuple)
            for token_idx in token_idx_list:
                new_key = []
                i = 0
                key, value = pre_token[token_idx]
                while i < len(key):
                    if i + 1 < len(key) and key[i : i + 2] == bytes_tuple_to_merge:
                        new_key.append(bytes_to_merge)
                        i += 1
                    else:
                        new_key.append(key[i])
                    i += 1

                new_key_tuple = tuple(new_key)
                modified_pre_token[token_idx] = (new_key_tuple, value)
                original_pre_token[token_idx] = (key, value)

            for token_idx, item in original_pre_token.items():
                key, value = item
                for i in range(len(key) - 1):
                    pairs[key[i : i + 2]][0] -= value
                    if token_idx in pairs[key[i : i + 2]][1]:
                        pairs[key[i : i + 2]][1].remove(token_idx)
                    if pairs[key[i : i + 2]][0] == 0:
                        del pairs[key[i : i + 2]]

            for token_idx, item in modified_pre_token.items():
                key, value = item
                pre_token[token_idx] = item
                for i in range(len(key) - 1):
                    if key[i : i + 2] not in pairs:
                        pairs[key[i : i + 2]] = [0, set()]
                    pairs[key[i : i + 2]][0] += value
                    pairs[key[i : i + 2]][1].add(token_idx)

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
        return list(pre_token.items())

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
        max_item = ((), [0, set()])
        for key, value in pairs.items():
            if value[0] > max_item[1][0] or (value[0] == max_item[1][0] and key > max_item[0]):
                max_item = (key, value)
        return max_item
