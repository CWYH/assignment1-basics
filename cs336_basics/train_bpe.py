"""Byte-level BPE tokenizer training."""

from __future__ import annotations

import os
import multiprocessing
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from heapq import heapify, heappop, heappush
from typing import TextIO

import regex


_PRETOKEN_PATTERN = regex.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


def _heap_entry(count: int, pair: tuple[bytes, bytes]) -> tuple:
    """Use negative keys to prioritize larger counts and byte pairs."""
    # The terminator is larger than any negated byte, so a longer token
    # precedes its prefix, e.g. b"aa" precedes b"a" at equal frequency.
    first_key = tuple(-byte for byte in pair[0]) + (1,)
    second_key = tuple(-byte for byte in pair[1]) + (1,)
    return -count, first_key, second_key, pair


def _iter_segments(source: TextIO, special_tokens: list[str], chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """Yield text between special tokens without splitting pretokens at read boundaries."""
    if not special_tokens:
        yield source.read()
        return
    pattern = regex.compile("|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True)))
    max_special_length = max(map(len, special_tokens))
    pending = ""
    while chunk := source.read(chunk_size):
        pending += chunk
        # Retain enough lookahead to recognize a longer overlapping special
        # token even if it straddles the next read boundary.
        safe_start_limit = len(pending) - max_special_length + 1
        consumed = 0
        for match in pattern.finditer(pending):
            if match.start() >= safe_start_limit:
                break
            yield pending[consumed : match.start()]
            consumed = match.end()
        pending = pending[consumed:]
    yield from pattern.split(pending)


def _document_ranges(input_path: str | os.PathLike, separator: bytes, chunk_size: int) -> list[tuple[int, int]]:
    """Move tentative byte offsets to the next complete document separator."""
    file_size = os.path.getsize(input_path)
    boundaries = {0, file_size}
    with open(input_path, "rb") as source:
        for offset in range(chunk_size, file_size, chunk_size):
            source.seek(offset)
            while block := source.read(max(65536, len(separator))):
                index = block.find(separator)
                if index >= 0:
                    boundaries.add(source.tell() - len(block) + index)
                    break
                if source.tell() == file_size:
                    break
                # A separator may straddle consecutive search blocks.
                source.seek(-(len(separator) - 1), os.SEEK_CUR)
    ordered = sorted(boundaries)
    return list(zip(ordered, ordered[1:]))


def _count_chunk(task: tuple[str | os.PathLike, int, int, str]) -> Counter[bytes]:
    input_path, start, end, special_token = task
    with open(input_path, "rb") as source:
        source.seek(start)
        text = source.read(end - start).decode("utf-8")
    # Match the universal-newline behavior of the sequential text reader.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    counts: Counter[bytes] = Counter()
    for segment in text.split(special_token):
        counts.update(match.group().encode("utf-8") for match in _PRETOKEN_PATTERN.finditer(segment))
    return counts


def _count_pretokens(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_workers: int,
    chunk_size: int = 32 * 1024 * 1024,
) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    # One unambiguous document delimiter permits independent byte ranges.
    # Keep small files sequential to avoid process-startup overhead.
    if num_workers > 1 and len(special_tokens) == 1 and os.path.getsize(input_path) > chunk_size:
        ranges = _document_ranges(input_path, special_tokens[0].encode("utf-8"), chunk_size)
        if len(ranges) > 1:
            tasks = [(input_path, start, end, special_tokens[0]) for start, end in ranges]
            with ProcessPoolExecutor(
                max_workers=min(num_workers, len(tasks)), mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                for partial_counts in pool.map(_count_chunk, tasks):
                    counts.update(partial_counts)
            return counts
    with open(input_path, encoding="utf-8") as source:
        for segment in _iter_segments(source, special_tokens):
            counts.update(match.group().encode("utf-8") for match in _PRETOKEN_PATTERN.finditer(segment))
    return counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_workers: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer on a text file.

    Args:
        input_path: Path to the training text file.
        vocab_size: Maximum vocabulary size, including initial bytes,
            merged tokens, and special tokens.
        special_tokens: Tokens to reserve in the vocabulary. Their occurrences
            form boundaries and are excluded from merge statistics.
        num_workers: Pretokenization worker limit; defaults to at most eight.
            Large files with one special token use document-aligned parallel
            counting. Other inputs use sequential counting. Scripts using
            multiprocessing must call training inside a __main__ guard.

    Returns:
        A vocabulary mapping token IDs to bytes, and a list of byte-pair
        merges in creation order.
    """
    special_tokens = list(dict.fromkeys(special_tokens))
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
    if num_workers < 1:
        raise ValueError("num_workers must be positive.")
    if any(not token for token in special_tokens):
        raise ValueError("Special tokens must not be empty.")
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must fit all 256 bytes and the special tokens.")

    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    merges: list[tuple[bytes, bytes]] = []
    if len(vocab) == vocab_size:
        return vocab, merges

    # Split before pretokenization so removed special tokens never join text.
    pretoken_counts = _count_pretokens(input_path, special_tokens, num_workers)

    # Store each distinct pretoken once, weighted by its corpus frequency.
    words = [tuple(vocab[byte] for byte in word) for word in pretoken_counts]
    frequencies = list(pretoken_counts.values())
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for word_id, word in enumerate(words):
        for pair, count in Counter(zip(word, word[1:])).items():
            pair_counts[pair] += count * frequencies[word_id]
            pair_words[pair].add(word_id)

    heap = [_heap_entry(count, pair) for pair, count in pair_counts.items()]
    heapify(heap)
    while len(vocab) < vocab_size and heap:
        negative_count, _, _, best_pair = heappop(heap)
        # Updated counts get new entries; discard obsolete entries lazily.
        if pair_counts.get(best_pair, 0) != -negative_count:
            continue

        merged_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_token
        merges.append(best_pair)
        changed_pairs: set[tuple[bytes, bytes]] = set()
        for word_id in list(pair_words[best_pair]):
            word = words[word_id]
            old_counts = Counter(zip(word, word[1:]))
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best_pair:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            words[word_id] = tuple(new_word)
            new_counts = Counter(zip(new_word, new_word[1:]))
            for pair in old_counts.keys() | new_counts.keys():
                delta = new_counts[pair] - old_counts[pair]
                if delta:
                    pair_counts[pair] += delta * frequencies[word_id]
                    changed_pairs.add(pair)
                if pair not in new_counts:
                    pair_words[pair].discard(word_id)
                elif pair not in old_counts:
                    pair_words[pair].add(word_id)

        for pair in changed_pairs:
            if pair_counts[pair] > 0:
                heappush(heap, _heap_entry(pair_counts[pair], pair))
            else:
                del pair_counts[pair]
                pair_words.pop(pair, None)

    return vocab, merges
