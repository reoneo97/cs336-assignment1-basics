from tests.adapters import run_train_bpe
import argparse
import resource

SPECIAL_TOKENS = ["<|endoftext|>"]
parser = argparse.ArgumentParser("Parser for training script")
parser.add_argument("dataset_path", type=str, help="Path for the dataset")
parser.add_argument("--vocab_size", type=int, help="Vocab Size for BPE Trainer")

args = parser.parse_args()

vocab, merges = run_train_bpe(args.dataset_path, args.vocab_size, SPECIAL_TOKENS)

max_len_token = max(vocab.values(), key=lambda x: len(x))
print(max_len_token)

peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KB, macOS: bytes
peak_mb = peak_kb / 1024  # on Linux
print(f"Peak memory: {peak_mb:.1f} MB")
