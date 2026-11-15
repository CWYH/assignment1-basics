"""Unprofiled Python baseline with the same stage boundaries as the native ports."""

import json
import sys
import time
from pathlib import Path

import cs336_basics.train_bpe as training


def main():
    input_path, output_dir, size, workers = sys.argv[1:]
    timeline = {}
    original_count = training._count_pretokens
    original_heapify = training.heapify

    def count(*args, **kwargs):
        result = original_count(*args, **kwargs)
        timeline["counted"] = time.perf_counter()
        return result

    def heapify(heap):
        original_heapify(heap)
        timeline["initialized"] = time.perf_counter()

    training._count_pretokens = count
    training.heapify = heapify
    start = time.perf_counter()
    vocab, merges = training.train_bpe(input_path, int(size), ["<|endoftext|>"], num_workers=int(workers))
    finish = time.perf_counter()
    # A vocabulary of 257 returns before preprocessing.
    counted = timeline.get("counted", finish)
    initialized = timeline.get("initialized", finish)
    metrics = {
        "pretoken_seconds": counted - start,
        "setup_seconds": initialized - counted,
        "merge_seconds": finish - initialized,
        "training_seconds": finish - start,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "vocab.tsv").write_text("".join(f"{i}\t{token.hex()}\n" for i, token in vocab.items()), encoding="utf-8")
    (output / "merges.tsv").write_text("".join(f"{a.hex()}\t{b.hex()}\n" for a, b in merges), encoding="utf-8")
    (output / "timings.json").write_text(json.dumps(metrics), encoding="utf-8")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
