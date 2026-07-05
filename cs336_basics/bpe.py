import regex as re
import os
from typing import BinaryIO


GPT2_PRETOKENIZE = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


class BPETrainer():
    def __init__(self, vocab_size: int, special_tokens:list[str]):
        assert vocab_size > 256 + len(special_tokens), "Vocab size too small to initialize"
        self.vocab_size = vocab_size
        self.merges = []
        self.special_tokens = []

        self.vocab = {i: bytes([i]) for i in range(256)}
        for tok in special_tokens:
            tok_bytes = tok.encode('utf-8')
            self.vocab[len(self.vocab)] = tok_bytes
            self.special_tokens.append(tok_bytes)

    def train_bpe(self, input_path: str | os.PathLike):

        with open(input_path, 'rb') as f:
        # TODO: Hardcoded to just take the first one 
            chunk_boundaries = find_chunk_boundaries(
                f, 8, self.special_tokens[0]
            )
        print(chunk_boundaries)

    def __repr__(self):
        return f'''
        Vocab Size: {self.vocab_size}
        Special Tokens: {self.special_tokens}
        VOCAB: {self.vocab}
        '''

class BPETokenizer():
    
    def __init__(self, special_tokens: list[str], concurrency: int):
        self.vocab = dict()
        self.special_tokens = special_tokens
        self.concurrency = concurrency
        # for sc in self.special_chars:
            # self.vocab[]
    def pretokenize(text):

        matches = re.finditer(GPT2_PRETOKENIZE, text)



    # def async_chunk():
        # pass

    def merge():
        pass
