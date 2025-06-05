from collections import defaultdict
import os
import regex as re
from dataclasses import dataclass, field
import multiprocessing
from .pretokenization_example import find_chunk_boundaries

# String pattern
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# bytes pattern
PAT = b"'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\\sa-zA-Z0-9]+|\\s+(?!\\S)|\\s+"


@dataclass
class PairValue:
    count: int = 0
    indices: set = field(default_factory=lambda: set())


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
        pre_token = self.pre_tokenize(input_path=input_path)
        num_merges = vocab_size - len(self.vocab)

        modified_pre_token = defaultdict(tuple)
        original_pre_token = defaultdict(tuple)

        pairs = defaultdict(PairValue)
        for idx, item in enumerate(pre_token):
            key, value = item
            for i in range(len(key) - 1):
                bytes_tuple = key[i : i + 2]
                pairs[bytes_tuple].count += value
                pairs[bytes_tuple].indices.add(idx)

        for iteration in range(num_merges):
            # print(f"iteration: {iteration}")
            max_item = self.find_max(pairs)
            bytes_tuple_to_merge = max_item[0]
            token_idx_list = max_item[1].indices

            self.merges.append(bytes_tuple_to_merge)
            bytes_to_merge = self.bytes_tuple_to_bytes(bytes_tuple_to_merge)
            self.vocab[len(self.vocab)] = bytes_to_merge

            modified_pre_token.clear()
            original_pre_token.clear()

            # Check the tokens affected
            for token_idx in token_idx_list:
                key, value = pre_token[token_idx]
                new_key = []
                i = 0
                while i < len(key):
                    if i + 1 < len(key) and key[i : i + 2] == bytes_tuple_to_merge:
                        new_key.append(bytes_to_merge)
                        i += 1
                    else:
                        new_key.append(key[i])
                    i += 1

                modified_pre_token[token_idx] = (tuple(new_key), value)
                original_pre_token[token_idx] = (key, value)

            # Remove the original tokens
            for token_idx, item in original_pre_token.items():
                key, value = item
                for i in range(len(key) - 1):
                    bytes_tuple = key[i : i + 2]
                    pairs[bytes_tuple].count -= value
                    pairs[bytes_tuple].indices.discard(token_idx)
                    if pairs[bytes_tuple].count == 0:
                        del pairs[bytes_tuple]

            # update the new tokens
            for token_idx, item in modified_pre_token.items():
                key, value = item
                pre_token[token_idx] = item
                for i in range(len(key) - 1):
                    bytes_tuple = key[i : i + 2]
                    pairs[bytes_tuple].count += value
                    pairs[bytes_tuple].indices.add(token_idx)

        return (self.vocab, self.merges)

    def initialize_special_tokens(
        self,
        special_tokens: list[str] | None = None,
    ):
        self.special_tokens = set()
        self.escaped_special_tokens = set()
        if special_tokens is None:
            self.delimiter = None
            return

        for s in special_tokens:
            self.special_tokens.add(s.encode("utf-8", errors="ignore"))
            self.escaped_special_tokens.add(re.escape(s).encode("utf-8", errors="ignore"))

        self.delimiter = b"|".join(self.escaped_special_tokens)

    def pre_tokenize(self, input_path):
        num_processes = multiprocessing.cpu_count() - 1
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
            tasks = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

            with multiprocessing.Pool(processes=num_processes) as pool:
                pre_token_list = pool.map(self.process_doc, tasks)

        pre_token = defaultdict(int)
        for pre_tok in pre_token_list:
            for key, value in pre_tok.items():
                pre_token[key] += value
        return list(pre_token.items())

    def process_doc(self, args):
        file_path, start, end = args
        pre_token = defaultdict(int)
        with open(file_path, 'rb') as f:
            f.seek(start)
            docs = f.read(end - start)
            docs_list = re.split(self.delimiter, docs)
            for doc in docs_list:
                for m in re.finditer(PAT, doc):
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
        max_item = ((), PairValue())
        for key, value in pairs.items():
            if value.count > max_item[1].count or (value.count == max_item[1].count and key > max_item[0]):
                max_item = (key, value)
        return max_item
