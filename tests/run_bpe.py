from .adapters import run_train_bpe
import pathlib
import pickle
from cs336_basics.bpe import BPETokenizer


DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
OUTPUT_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "outputs"


tx = "tiny"
if tx == "owt":
    input_path = DATA_PATH / "owt_train.txt"
    vocab_path = OUTPUT_PATH / "owt_vocab.pkl"
    merges_path = OUTPUT_PATH / "owt_merges.pkl"
    vocab_size = 32000
else:
    input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    vocab_path = OUTPUT_PATH / "tiny_stories_vocab.pkl"
    merges_path = OUTPUT_PATH / "tiny_stories_merges.pkl"
    vocab_size = 10000

tokenizer = BPETokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])

with open(input_path, "r") as f:
    ids = tokenizer.encode(f.read())
    print(len(ids))
