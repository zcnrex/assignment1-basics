from collections import defaultdict
import os
import regex as re
from .utils import print_
import time
from dataclasses import dataclass, field

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# TODO Figure out pattern match for bytes (or a correct way to convert to str before re)
PAT = b"'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\\sa-zA-Z0-9]+|\\s+(?!\\S)|\\s+"
PATTERN = "d"


@dataclass
class PairValue:
    count: int = 0
    indices: dict = defaultdict(int)
    indices: defaultdict = field(default_factory=lambda: defaultdict(int))


class BPETokenizerTrainer:
    def __init__(
        self,
    ):
        self.vocab = defaultdict(bytes)
        self.merges = []  # list[tuple[bytes, bytes]],
        self.special_tokens = set()

    def train(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], **kwargs):
        time1 = time.time()
        self.initialize_special_tokens(special_tokens=special_tokens)
        time2 = time.time()
        # print_(f"TIME initialize_special_tokens: {time2 - time1}", 2)
        self.initialize_vocab()
        time3 = time.time()
        # print_(f"TIME initialize_vocab: {time3 - time2}", 2)
        count = 0
        pre_token = self.pretokenize(input_path=input_path, pattern=PATTERN)  # [((bytes, ...), count)]
        print_("pre_token ", pre_token, 3)
        time4 = time.time()
        # print_(f"TIME pretokenize: {time4 - time3}", 2)
        num_merge = vocab_size - len(self.vocab)
        curr_merge = 0
        pairs = defaultdict(list)  # {(bytes, bytes): [count, {(token_idx, count_in_token)}]}
        for idx, item in enumerate(pre_token):
            key, value = item
            for i in range(len(key) - 1):
                if key[i : i + 2] not in pairs:
                    pairs[key[i : i + 2]] = [0, dict()]
                # pairs[key[i : i + 2]].count += value
                pairs[key[i : i + 2]][0] += value
                if idx not in pairs[key[i : i + 2]][1]:
                    pairs[key[i : i + 2]][1][idx] = 0
                pairs[key[i : i + 2]][1][idx] += 1
                # pairs[key[i : i + 2]].indices[idx] += 1

        # sorted_pairs = dict(sorted(pairs.items(), key=lambda x: (x[1][0], x[0]), reverse=True))
        print_("pairs ", pairs, 3)
        pairs_time = 0
        sort_time = 0
        new_token_time = 0
        while curr_merge < num_merge:
            print_("=========iter ", curr_merge, 3)
            print_("pre_token ", pre_token, 3)
            print_("pairs ", pairs, 3)
            time5_1 = time.time()
            curr_merge += 1
            max_item = self.find_max(pairs=pairs)
            pairs.pop(max_item[0])
            bytes_tuple_to_merge = max_item[0]
            # Pop the first item
            token_idx_byte_idx = max_item[1]
            time5_2 = time.time()
            pairs_time += time5_2 - time5_1
            # print_("pre_token", pre_token, 1)
            # if  max_item[0] == (b'n', b'd'):
            #     print_("max_item", max_item, 6)
            #     print_("bytes_tuple_to_merge ", bytes_tuple_to_merge, 6)
            #     print_("token_idx_byte_idx ", token_idx_byte_idx, 6)
            # sorted_pairs = sorted(pairs.items(), key=lambda x: (x[1], x[0]), reverse=True)
            time5_3 = time.time()
            sort_time += time5_3 - time5_2
            # print_(f"sorted_pairs {sorted_pairs}", 1)
            self.merges.append(bytes_tuple_to_merge)
            bytes_to_merge = self.bytes_tuple_to_bytes(bytes_tuple_to_merge)
            self.vocab[len(self.vocab)] = bytes_to_merge
            print_("bytes_to_merge: ", bytes_to_merge, 3)
            print_("bytes_tuple_to_merge: ", bytes_tuple_to_merge, 3)
            if bytes_tuple_to_merge in pairs:
                print_("pairs[bytes_tuple_to_merge]: ", pairs[bytes_tuple_to_merge], 3)
            for idx in token_idx_byte_idx[1].keys():
                byte_count_tuple = pre_token[idx]
                byte_tuple = byte_count_tuple[0]
                byte_count = byte_count_tuple[1]
                new_byte_list = []
                if idx == 449:
                    print("idx 449================")
                    print(f"byte_count_tuple {byte_count_tuple}")
                    print(f"bytes_to_merge {bytes_to_merge}")
                    print_("max_item", max_item, 6)
                i = 0
                while i < len(byte_tuple):
                    if i < len(byte_tuple) - 1 and (byte_tuple[i], byte_tuple[i + 1]) == bytes_tuple_to_merge:
                        if i > 0:
                            pair_to_update = (byte_tuple[i - 1], byte_tuple[i])
                            # new_pair = (byte_tuple[i - 1], bytes_to_merge)
                            new_pair = (new_byte_list[-1], bytes_to_merge)
                            self.update_pairs(pairs, pair_to_update, new_pair, idx, byte_count, byte_count_tuple)
                        if i < len(byte_tuple) - 2:
                            pair_to_update = (byte_tuple[i + 1], byte_tuple[i + 2])
                            new_pair = (bytes_to_merge, byte_tuple[i + 2])
                            self.update_pairs(pairs, pair_to_update, new_pair, idx, byte_count, byte_count_tuple)
                        new_byte_list.append(bytes_to_merge)
                        i += 1
                    else:
                        new_byte_list.append(byte_tuple[i])
                    i += 1
                pre_token[idx] = (tuple(new_byte_list), byte_count)

        time5 = time.time()
        # print_(f"TIME whole while: {time5 - time4}", 6)
        # print_(f"pairs_time {pairs_time}", 6)
        # print_(f"sort_time {sort_time}", 6)
        # print_(f"new_token_time {new_token_time}", 6)

        # print_(f"count: {count}", 1)
        print_("self.vocab: ", self.vocab, 3)
        print_("self.merges: ", self.merges, 3)
        return (self.vocab, self.merges)

    def update_pairs(self, pairs, pair_to_update, new_pair, idx, byte_count, byte_count_tuple):
        # if pair_to_update == (b'n', b'd') and pair_to_update in pairs:
        # #     print(f"pair_to_update {pair_to_update}")
        #     print("line118 =========")
        #     print(f"pairs[pair_to_update] {pairs[pair_to_update]}")
        #     print(f"byte_count {byte_count}")
        if idx == 449:
            print(f"valueerror============")
            print(f"self.merges {self.merges}")
            print(f"pair_to_update {pair_to_update}")
            print(f"new_pair {new_pair}")
            print(f"byte_count_tuple {byte_count_tuple}")
            print(f"pairs[pair_to_update] {pairs[pair_to_update]}")
            # print(f"pairs[new_pair] {pairs[new_pair]}")
            print(pair_to_update == (b"n", b"d"))
        if new_pair == (b"in", b"in"):
            breakpoint()
        if pair_to_update not in pairs:
            raise ValueError()
        if pair_to_update in pairs and pairs[pair_to_update][0] >= byte_count:
            pairs[pair_to_update][0] -= byte_count
            if idx in pairs[pair_to_update][1]:
                pairs[pair_to_update][1][idx] -= 1
                if pairs[pair_to_update][1][idx] == 0:
                    del pairs[pair_to_update][1][idx]

            if pairs[pair_to_update][0] == 0:
                if pair_to_update == (b"n", b"d"):
                    print("line128===============")
                    print(f"pair_to_update {pair_to_update}")
                    print(f"pairs[pair_to_update] {pairs[pair_to_update]}")
                    print(f"byte_count {byte_count}")
                    print("============pair_to_update == (b'n', b'd')")
                    print(f"pairs[pair_to_update] {pairs[pair_to_update]}")
                    print("line132===============")
                pairs.pop(pair_to_update)

        if new_pair not in pairs:
            pairs[new_pair] = [0, dict()]
        pairs[new_pair][0] += byte_count
        if idx not in pairs[new_pair][1]:
            pairs[new_pair][1][idx] = 0
        pairs[new_pair][1][idx] += 1

    def initialize_special_tokens(
        self,
        special_tokens: list[str] | None = None,
    ):
        self.special_tokens = set()
        if special_tokens is None:
            return
        for s in special_tokens:
            self.special_tokens.add(s.encode("utf-8", errors="ignore"))

    def pretokenize(self, input_path, pattern="ws") -> list[tuple[tuple[bytes], int]]:
        pre_token = defaultdict(int)
        with open(input_path, "rb") as f:
            lines = f.readlines()
            for line in lines:
                if pattern == "ws":
                    tokens = line.strip(b"\n").split(b" ")
                    for token in tokens:
                        pre_token[self.bytes_to_bytes_tuple(token)] += 1
                else:
                    for m in re.finditer(PAT, line):
                        # pre_token[m.group(0).encode("utf-8")] += 1
                        pre_token[self.bytes_to_bytes_tuple(m.group(0))] += 1
                # if len(pre_token) > 100:
                #     break
        return list(pre_token.items())

    def initialize_vocab(self):
        for i, t in enumerate(self.special_tokens):
            self.vocab[i] = t

        num_special_tokens = len(self.special_tokens)
        for i in range(256):
            self.vocab[i + num_special_tokens] = bytes([i])

    def bytes_to_bytes_list(self, input_bytes):
        bytes_list = []
        for i in input_bytes:
            bytes_list.append(bytes([i]))
        return bytes_list

    def bytes_to_bytes_tuple(self, input_bytes):
        return tuple(self.bytes_to_bytes_list(input_bytes))

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
