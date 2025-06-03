from collections import defaultdict
import os
import regex as re
from .utils import print_
import time

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# TODO Figure out pattern match for bytes (or a correct way to convert to str before re)
PAT = b"'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\\sa-zA-Z0-9]+|\\s+(?!\\S)|\\s+"


class BPETokenizerTrainer:
    def __init__(
        self,
    ):
        self.vocab = defaultdict(bytes)
        self.merges = []
        self.special_tokens = set()

    def train(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], **kwargs):
        time1 = time.time()
        self.initialize_special_tokens(special_tokens=special_tokens)
        time2 = time.time()
        print_(f"TIME initialize_special_tokens: {time2 - time1}", 2)
        self.initialize_vocab()
        time3 = time.time()
        print_(f"TIME initialize_vocab: {time3 - time2}", 2)
        count = 0
        pre_token = self.pretoke(input_path=input_path, pattern="d")
        time4 = time.time()
        print_(f"TIME pretoke: {time4 - time3}", 2)
        num_merge = vocab_size - len(self.vocab)
        curr_merge = 0

        pairs_time = 0
        sort_time = 0
        new_token_time = 0
        while curr_merge < num_merge:
            time5_1 = time.time()
            curr_merge += 1
            pairs = defaultdict(int)
            for key, value in pre_token.items():
                for i in range(len(key) - 1):
                    pairs[key[i : i + 2]] += value
            time5_2 = time.time()
            pairs_time += time5_2 - time5_1
            print_(f"=========iter {curr_merge} =========", 1)
            print_(f"pre_token {pre_token}", 1)
            print_(f"pairs {pairs}", 1)
            sorted_pairs = sorted(pairs.items(), key=lambda x: (x[1], x[0]), reverse=True)
            time5_3 = time.time()
            sort_time += time5_3 - time5_2
            print_(f"sorted_pairs {sorted_pairs}", 1)
            bytes_tuple_to_merge = sorted_pairs[0][0]
            self.merges.append(bytes_tuple_to_merge)
            bytes_to_merge = self.bytes_tuple_to_bytes(bytes_tuple_to_merge)
            self.vocab[len(self.vocab)] = bytes_to_merge
            print_(f"bytes_to_merge: {bytes_to_merge}", 1)
            new_pre_token = defaultdict(int)
            for key, value in pre_token.items():
                new_key = []
                i = 0
                while i < len(key):
                    if i + 1 < len(key) and key[i : i + 2] == bytes_tuple_to_merge:
                        new_key.append(bytes_to_merge)
                        i += 1
                    else:
                        new_key.append(key[i])
                    i += 1
                new_pre_token[tuple(new_key)] = value
            time5_4 = time.time()
            new_token_time += time5_4 - time5_3
            print_(f"new_pre_token: {new_pre_token}", 1)
            pre_token = new_pre_token

        time5 = time.time()
        print_(f"TIME whole while: {time5 - time4}", 2)
        print_(f"pairs_time {pairs_time}", 2)
        print_(f"sort_time {sort_time}", 2)
        print_(f"new_token_time {new_token_time}", 2)

        print_(f"count: {count}", 1)
        print_(f"self.vocab: {self.vocab}", 1)
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
        print_(self.special_tokens, 1)

    def pretoke(self, input_path, pattern="ws"):
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
                        # pre_token[m.group(0).encode("utf-8")] += 1
                        pre_token[self.bytes_to_bytes_tuple(m.group(0))] += 1
        return pre_token

    def initialize_vocab(self):
        for i, t in enumerate(self.special_tokens):
            self.vocab[i] = t

        num_special_tokens = len(self.special_tokens)
        for i in range(256):
            self.vocab[i + num_special_tokens] = bytes([i])

        # print_(f"init vocab {self.vocab}", 1)

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
