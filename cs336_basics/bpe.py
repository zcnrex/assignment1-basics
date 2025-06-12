import pickle
from collections.abc import Iterable, Iterator
import regex as re
import time
import numpy as np

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
        self.merge_results = {merge: self.bytes_tuple_to_bytes(merge) for merge in self.merges}
        self.initialize_special_tokens(special_tokens=special_tokens)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        b_texts = text.encode("utf-8")
        encoded_token = []
        if self.delimiter:
            b_text_list = re.split(self.delimiter, b_texts)
        else:
            b_text_list = [b_texts]
        for b_text in b_text_list:
            encoded_token.extend(self.process_text(b_text=b_text))
        return encoded_token

    def encode_chunk(self, task):
        idx, start, end, input_path, root_output_path = task

        print(f"[INFO] {self.readable_time()} chunk {idx} start")
        with open(input_path, "rb") as f:
            f.seek(start)
            docs = f.read(end - start)
            print(f"[INFO] {self.readable_time()} chunk {idx} docs read")
            docs_list = re.split(self.delimiter, docs)
            print(f"[INFO] {self.readable_time()} chunk {idx} docs split")
            self.process_and_save_numpy(docs_list=docs_list, root_output_path=root_output_path, idx=idx)

    def process_and_save_numpy(self, docs_list, root_output_path, idx):
        out_list = []
        output_path = root_output_path / (str(idx) + ".npy")
        print(f"[INFO] num docs {len(docs_list)} for chunk {idx}")
        for i, b_text in enumerate(docs_list):
            if b_text in self.special_tokens:
                out_list.append(self.inverse_vocab[b_text])
            else:
                for m in re.finditer(PAT, b_text):
                    tokens = self._tokenize(self.bytes_to_bytes_tuple(m.group(0)))
                    out_list.extend([self.inverse_vocab[token] for token in tokens])
            if i % 100 == 0:
                print(
                    f"[INFO] {self.readable_time()} Processed {len(out_list):,} tokens (at doc index {i}) for chunk {idx}"
                )

        print(
            f"[INFO] {self.readable_time()} Done processing all tokens for chunk {idx}, total output token count {len(out_list)}"
        )
        arr = np.array(out_list, dtype=np.uint32)
        print(f"[INFO] {self.readable_time()} Done creating np array for chunk {idx}")
        np.save(output_path, arr)
        print(f"[DONE] {self.readable_time()} Saved {len(arr):,} tokens to {output_path}")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        output = []
        for id in ids:
            output.extend(list(self.vocab[id]))
        return bytes(output).decode(encoding="utf-8", errors="ignore")

    def process_text(self, b_text):
        if b_text in self.special_tokens:
            return [self.inverse_vocab[b_text]]
        else:
            encoded_token = []
            for m in re.finditer(PAT, b_text):
                self._encode(self.bytes_to_bytes_tuple(m.group(0)), encoded_token)
            return encoded_token

    def _encode(self, token_tuple, encoded_token):
        new_token_list = self._tokenize(token_tuple=token_tuple)
        for new_token in new_token_list:
            encoded_token.append(self.inverse_vocab[new_token])
        return encoded_token

    def _tokenize(self, token_tuple):
        tokens = list(token_tuple)
        new_token_set = set(zip(tokens, tokens[1:]))

        token_len = len(tokens)
        if token_len < 2:
            return tokens

        for merge in self.merges:
            if merge not in new_token_set:
                continue

            write_idx = 0
            read_idx = 0

            while read_idx < token_len:
                if read_idx + 1 < token_len and merge[0] == tokens[read_idx] and merge[1] == tokens[read_idx + 1]:
                    if read_idx + 2 < token_len:
                        new_token_set.add((self.merge_results[merge], tokens[read_idx + 2]))

                    tokens[write_idx] = self.merge_results[merge]

                    if read_idx > 0:
                        new_token_set.add((tokens[write_idx - 1], self.merge_results[merge]))

                    read_idx += 2
                else:
                    tokens[write_idx] = tokens[read_idx]
                    read_idx += 1

                write_idx += 1

            token_len = write_idx
            if token_len < 2:
                return tokens[:token_len]

        return tokens[:token_len]

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
        self.escaped_special_tokens = []
        self.delimiter = b""
        if special_tokens is None:
            return
        for s in special_tokens:
            self.special_tokens.add(s.encode("utf-8", errors="ignore"))
            self.escaped_special_tokens.append(re.escape(s).encode("utf-8", errors="ignore"))
        self.escaped_special_tokens.sort(key=lambda x: len(x), reverse=True)
        self.delimiter = b"(" + b"|".join(self.escaped_special_tokens) + b")"

    def readable_time(self):
        timestamp = time.time()
        local_time = time.localtime(timestamp)
        return time.strftime("%Y-%m-%d %H:%M:%S", local_time)
