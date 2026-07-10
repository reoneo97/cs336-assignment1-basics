import regex as re
import os
from typing import BinaryIO
from collections.abc import Iterator, Iterable
from collections import Counter, defaultdict
from multiprocessing import Pool
import time
from pathlib import Path
import json

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

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


class BPETrainer:
    def __init__(self, vocab_size: int, special_tokens: list[str], parallel: int = 4):
        assert vocab_size > 256 + len(special_tokens), "Vocab size too small to initialize"
        self.merges = []
        self.special_tokens: list[str] = []
        self.parallel = parallel
        self.vocab_size = vocab_size

        self.vocab = {i: bytes([i]) for i in range(256)}
        for tok in special_tokens:
            tok_bytes = tok.encode("utf-8")
            self.vocab[len(self.vocab)] = tok_bytes
            self.special_tokens.append(tok)

    def train_bpe(self, input_path: str | os.PathLike):
        # Parallel Pre-tokenization
        st = time.time()
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.parallel, self.special_tokens[0].encode("utf-8"))
            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks.append(chunk)
            with Pool(processes=self.parallel) as pool:
                counters = pool.map(self.pretokenize, chunks)
            counter = sum(counters, Counter())

        print("Pretokenization took", time.time() - st, "seconds")
        st = time.time()
        # --- Word-ID based structures ---
        # This was needed the token tuples will change over time. word_seqs basically
        # keeps track of all pretokenized words to index and update pair_counts/locations
        word_seqs: list[list[int]] = []
        pair_counts = defaultdict(int)
        pair_locations = defaultdict(set)

        for k, v in counter.items():
            wid = len(word_seqs)
            word_seqs.append(list(k))
            for i in range(len(k) - 1):
                pair = (k[i], k[i + 1])
                pair_counts[pair] += v
                pair_locations[pair].add(wid)

        while len(self.vocab) < self.vocab_size:
            # Lexical Ordering Tie Breaker
            tid1, tid2 = max(
                pair_counts,
                key=lambda p: (pair_counts[p], self.vocab[p[0]], self.vocab[p[1]]),
            )
            tok1, tok2 = self.vocab[tid1], self.vocab[tid2]
            self.merges.append((tok1, tok2))

            self.vocab[len(self.vocab)] = tok1 + tok2
            new_tid = len(self.vocab) - 1

            wids = pair_locations[(tid1, tid2)].copy()

            for wid in wids:
                seq = word_seqs[wid]
                freq = counter[self.decode(seq)]
                L = len(seq)
                i = 0
                new_seq = []

                while i < L:
                    if i < L - 1 and seq[i] == tid1 and seq[i + 1] == tid2:
                        if new_seq:
                            left = new_seq[-1]
                            pair_counts[(left, tid1)] -= freq
                            pair_counts[(left, new_tid)] += freq

                        new_seq.append(new_tid)

                        if i + 2 < L:
                            pair_counts[(tid2, seq[i + 2])] -= freq
                            pair_counts[(new_tid, seq[i + 2])] += freq

                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1

                # Reconcile pair_locations based on DISTINCT pairs present before/after,
                # not per-occurrence — a pair can appear more than once in a word, and
                # discard/add per-match can desync the location set from reality.
                old_pairs = {(seq[j], seq[j + 1]) for j in range(len(seq) - 1)}
                new_pairs = {(new_seq[j], new_seq[j + 1]) for j in range(len(new_seq) - 1)}

                for pair in old_pairs - new_pairs:
                    pair_locations[pair].discard(wid)
                for pair in new_pairs:
                    pair_locations[pair].add(wid)

                word_seqs[wid] = new_seq

            del pair_counts[(tid1, tid2)]
            del pair_locations[(tid1, tid2)]
        print("Merges took", time.time() - st, "seconds")
        return self.vocab, self.merges

    def pretokenize(self, chunk: str):
        splitter = "|".join([re.escape(tok) for tok in self.special_tokens])
        sub_chunks = re.split(splitter, chunk)

        return Counter(
            tok.group().encode("utf-8") for sub_chunk in sub_chunks for tok in re.finditer(GPT2_PRETOKENIZE, sub_chunk)
        )

    def __repr__(self):
        return f"""
        Vocab Size: {len(self.vocab)}
        Special Tokens: {self.special_tokens}
        VOCAB: {self.vocab}
        """

    def decode(self, tokens: tuple[int]):
        return b"".join([self.vocab[i] for i in tokens])


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab.copy()
        self.merges = merges

        if special_tokens is None:
            special_tokens = []
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        for token in special_tokens:
            encoded_token = token.encode("utf-8")
            if encoded_token not in self.vocab.values():
                self.vocab[len(self.vocab)] = encoded_token

        self.special_tokens = special_tokens
        self.vocab_mapper: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        vocab_filepath = Path(vocab_filepath)
        assert vocab_filepath.suffix == ".json", "Vocab should be in the json format"
        with open(vocab_filepath) as f:
            vocab = json.load(f)

        with open(merges_filepath, encoding="utf-8") as f:
            merges = []
            for line in f:
                s1, s2 = line.rstrip().split(" ")
                merges.append((s1.encode("utf-8"), s2.encode("utf-8")))
        return cls(vocab, merges, special_tokens)

    def chunk_by_special_tokens(
        self,
        text: str,
    ) -> list[bytes]:
        """
        Chunk by special tokens to ensure that the special tokens are kept and 
        not converted to bytes. Regex is slightly different with the () so that
        special tokens will be in its own chunk - This is so that special tokens
        can be tokenized independently 
        """
        if self.special_tokens:
            splitter = "|".join([re.escape(tok) for tok in self.special_tokens])
            chunks = re.split(f"({splitter})", text)
        else:
            chunks = [text]
        return chunks

    def pretokenize(self, chunk: str) -> list[list[bytes]]:
        return [
            [bytes([byte]) for byte in token.group().encode("utf-8")] for token in re.finditer(GPT2_PRETOKENIZE, chunk)
        ]

    def encode_text(self, text: str) -> list[int]:
        """
        Helper class to encode a string that does not contain special characters.
        pretokens are a list of byte objects, initially the objects are just single bytes. 
        As we iterate over merges, we combine the bytes to form larger strings
        After all possible merges are done, tokenize them based on the vocab mapper
        
        Can definitely be paralellized in the future, since each tokenization of each 
        pretoken is done independently
        """
        pretokens = self.pretokenize(text)
        for b1, b2 in self.merges:
            for pt_i in range(len(pretokens)):
                new = []
                pretoken = pretokens[pt_i]  # List of bytes
                i = 0
                while i < len(pretoken):
                    if i < len(pretoken) - 1 and pretoken[i] == b1 and pretoken[i + 1] == b2:
                        new.append(b1 + b2)
                        i += 2
                    else:
                        new.append(pretoken[i])
                        i += 1
                pretokens[pt_i] = new

        tokenized = []
        for pretoken in pretokens:
            for token_b in pretoken:
                tokenized.append(
                    self.vocab_mapper[token_b]  # No fallback because it should always be present
                )

        return tokenized

    def encode(self, text: str) -> list[int]:
        chunks = self.chunk_by_special_tokens(text)
        res = []
        for chunk in chunks:
            if not len(chunk):
                continue
            elif self.special_tokens and chunk in self.special_tokens:
                # Check if the chunk corresponds to a special token
                res.append(self.vocab_mapper[chunk.encode("utf-8")])
            else:
                # No special token, tokenize as normal 
                res += self.encode_text(chunk)
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_str = b"".join([self.vocab[i] for i in ids])
        return byte_str.decode("utf-8", errors="replace")
