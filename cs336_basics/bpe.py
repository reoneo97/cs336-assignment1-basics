import regex as re
import os
from typing import BinaryIO
from collections import Counter, defaultdict
from multiprocessing import Pool

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
    def __init__(self,
                 vocab_size: int,
                 special_tokens:list[str],
                 parallel: int = 4
        ):
        assert vocab_size > 256 + len(special_tokens), "Vocab size too small to initialize"
        self.merges = []
        self.special_tokens: list[str] = []
        self.parallel = parallel
        self.vocab_size = vocab_size

        self.vocab = {i: bytes([i]) for i in range(256)}
        for tok in special_tokens:
            tok_bytes = tok.encode('utf-8')
            self.vocab[len(self.vocab)] = tok_bytes
            self.special_tokens.append(tok)

    def train_bpe(self, input_path: str | os.PathLike):

        # Parallel Pre-tokenization
        with open(input_path, 'rb') as f:
            # TODO: Hardcoded to just take the first one
            boundaries = find_chunk_boundaries(
                f, self.parallel, self.special_tokens[0].encode('utf-8')
            )
            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks.append(chunk)
            with Pool(processes=self.parallel) as pool:
                counters = pool.map(self.pretokenize, chunks)
            counter = sum(counters, Counter())

        # Compute Merges
        # print(counter)
        merges =[]
        pair_counts = defaultdict(int)
        pair_locations = defaultdict(set)
        for k,v in counter.items():
            for i in range(len(k) - 1): 
                # No need to check vocab since this is pre-merge, int will be directly mapped to vocab
                pair = (k[i], k[i+1])
                pair_counts[pair] += v
                pair_locations[pair].add(tuple(k))

        while len(self.vocab) < self.vocab_size:
            tid1, tid2 = max(
                pair_counts,
                key=lambda p: (pair_counts[p], self.vocab[p[0]], self.vocab[p[1]])
            )
            tok1, tok2 = self.vocab[tid1], self.vocab[tid2]
            self.merges.append((tok1, tok2))

            self.vocab[len(self.vocab)] = tok1+tok2
            new_tid = len(self.vocab) - 1
            locations = pair_locations[(tid1, tid2)].copy()

            for location in locations:
                counts = counter[self.decode(location)]
                l = len(location)
                i = 0
                new_location = []
                pairs_touched = []
                while i < l:
                    if i < l - 1 and location[i] == tid1 and location[i+1] == tid2:
                        # LEFT neighbor: read off new_location (already-merged state), not location[i-1]
                        if new_location:  # equivalent to "i > 0" but correct after merges
                            left = new_location[-1]
                            pair_counts[(left, tid1)] -= counts
                            pair_counts[(left, new_tid)] += counts
                            pair_locations[(left, tid1)].discard(location)
                            pairs_touched.append((left, new_tid))

                        new_location.append(new_tid)

                        # RIGHT neighbor: location[i+2] is still safe to read directly here
                        if i + 2 < l:
                            pair_counts[(tid2, location[i+2])] -= counts
                            pair_counts[(new_tid, location[i+2])] += counts
                            pair_locations[(tid2, location[i+2])].discard(location)
                            pairs_touched.append((new_tid, location[i+2]))

                        i += 2
                    else:
                        new_location.append(location[i])
                        i += 1
                new_location = tuple(new_location)
                for pair in pairs_touched:
                    pair_locations[pair].add(new_location)

            del pair_counts[(tid1, tid2)]
            del pair_locations[(tid1, tid2)]
        # for merge in self.merges:
        # print(merge)
        return self.vocab, self.merges

    def pretokenize(
            self, chunk: str):
        counter = dict()
        splitter = '|'.join([re.escape(tok) for tok in self.special_tokens])
        sub_chunks = re.split(splitter, chunk)

        return Counter(
            tok.group().encode('utf-8') for sub_chunk in sub_chunks for tok in re.finditer(GPT2_PRETOKENIZE, sub_chunk)
        )

    def __repr__(self):
        return f'''
        Vocab Size: {len(self.vocab)}
        Special Tokens: {self.special_tokens}
        VOCAB: {self.vocab}
        '''
    
    def decode(self, tokens: tuple[int]):
        return b"".join([self.vocab[i] for i in tokens])


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
