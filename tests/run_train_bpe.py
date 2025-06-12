from .adapters import run_train_bpe
import pathlib
import pickle

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
OUTPUT_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "outputs"

tx = "owt"
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


vocab, merges = run_train_bpe(
    input_path=input_path,
    vocab_size=vocab_size,
    special_tokens=["<|endoftext|>"],
)

with open(vocab_path, "wb") as f:
    pickle.dump(vocab, f)

# Save merges to text file
with open(merges_path, "wb") as f:
    pickle.dump(merges, f)


# with open("outputs/tiny_stories_merges.pkl", 'rb') as f:
#     loaded_merges = pickle.load(f)
