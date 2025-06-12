from .adapters import run_train_bpe
import pathlib
import pickle
from cs336_basics.bpe import BPETokenizer
import multiprocessing
from cs336_basics.pretokenization_example import find_chunk_boundaries


DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
OUTPUT_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "outputs"


tx = "owt"
if tx == "owt":
    input_path = DATA_PATH / "owt_train.txt"
    vocab_path = OUTPUT_PATH / "owt_vocab.pkl"
    merges_path = OUTPUT_PATH / "owt_merges.pkl"
    token_id_path = OUTPUT_PATH / "owt_ids"
    vocab_size = 32000
else:
    input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    vocab_path = OUTPUT_PATH / "tiny_stories_vocab.pkl"
    merges_path = OUTPUT_PATH / "tiny_stories_merges.pkl"
    token_id_path = OUTPUT_PATH / "tiny_stories_ids"
    vocab_size = 10000

tokenizer = BPETokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])


num_processes = multiprocessing.cpu_count() - 1
with open(input_path, "rb") as f:
    boundaries = find_chunk_boundaries(f, num_processes * 4, "<|endoftext|>".encode("utf-8"))
    tasks = []
    b_zip = zip(boundaries[:-1], boundaries[1:])
    for i, b in enumerate(b_zip):
        start, end = b
        tasks.append((i, start, end, input_path, token_id_path))

    with multiprocessing.Pool(processes=num_processes) as pool:
        pre_token_list = pool.map(tokenizer.encode_chunk, tasks)

# tiny 2,227,753,162 => 200,000,000 per process, 15s per 1 million => 50min
# owt 11,920,511,059
