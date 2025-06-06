from collections import defaultdict
import pickle
from collections.abc import Iterable, Iterator
import regex as re

PAT = b"'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\\sa-zA-Z0-9]+|\\s+(?!\\S)|\\s+"


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.initialize_special_tokens(special_tokens=special_tokens)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pre_token = self.pre_tokenize(text)
        encoded_token = []
        for token_tuple in pre_token:
            original_token_list = list(token_tuple)
            while True:
                new_token_list = original_token_list
                for merge in self.merges:
                    idx = 0
                    tmp_token_list = []
                    while idx < len(new_token_list):
                        if idx < len(new_token_list) - 1 and merge == (new_token_list[idx], new_token_list[idx + 1]):
                            tmp_token_list.append(self.bytes_tuple_to_bytes((new_token_list[idx], new_token_list[idx + 1])))
                            idx += 1
                        else:
                            tmp_token_list.append(new_token_list[idx])
                        idx += 1
                    new_token_list = tmp_token_list
                if len(new_token_list) == len(original_token_list):
                    break
                else:
                    original_token_list = new_token_list
            for new_token in new_token_list:
                encoded_token.append(self.inverse_vocab[new_token])
        return encoded_token

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        output = []
        for id in ids:
            output.extend(list(self.vocab[id]))
        # breakpoint()
        return bytes(output).decode(encoding="utf-8", errors="ignore")

    def pre_tokenize(self, text):
        b_texts = text.encode("utf-8")
        pre_token = []
        if self.delimiter:
            b_text_list = re.split(self.delimiter, b_texts)
        else:
            b_text_list = [b_texts]
        for i, b_text in enumerate(b_text_list):
            if i % 2 == 0:
                for m in re.finditer(PAT, b_text):
                    pre_token.append(self.bytes_to_bytes_tuple(m.group(0)))
            else:
                pre_token.append((b_text,))
        return pre_token

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

    def initialize_special_tokens(
        self,
        special_tokens: list[str] | None = None,
    ):
        self.special_tokens = set()
        self.escaped_special_tokens = set()
        self.delimiter = b""
        if special_tokens is None:
            return
        for s in special_tokens:
            self.special_tokens.add(s.encode("utf-8", errors="ignore"))
            self.escaped_special_tokens.add(re.escape(s).encode("utf-8", errors="ignore"))
        self.delimiter = b"(" + b"|".join(self.escaped_special_tokens) + b")"
