from collections import defaultdict
from .utils import print_


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab if vocab else defaultdict(bytes)
        self.merges = merges if merges else []
        self.initialize_special_tokens(special_tokens=special_tokens)

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
