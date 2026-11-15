"""Run and measure a BPE training experiment; serialize bytes losslessly as hex."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import threading
import time
from pathlib import Path

import psutil

from cs336_basics.train_bpe import train_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--special-token", action="append", default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    special_tokens = args.special_token if args.special_token is not None else ["<|endoftext|>"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    with args.input_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    baseline = process.memory_info().rss
    cpu_start = process.cpu_times()
    stop_monitor = threading.Event()
    tree_peak_rss = [baseline]

    def monitor_memory() -> None:
        while not stop_monitor.is_set():
            total_rss = 0
            for member in [process, *process.children(recursive=True)]:
                try:
                    total_rss += member.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            tree_peak_rss[0] = max(tree_peak_rss[0], total_rss)
            stop_monitor.wait(0.05)

    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()
    print(f"Training on {args.input_path} ({args.input_path.stat().st_size:,} bytes)", flush=True)
    start = time.perf_counter()
    try:
        vocab, merges = train_bpe(args.input_path, args.vocab_size, special_tokens, num_workers=args.num_workers)
    finally:
        stop_monitor.set()
        monitor.join()
    elapsed = time.perf_counter() - start
    memory = process.memory_info()
    cpu_end = process.cpu_times()

    # Windows exposes an OS-maintained peak resident working set, including
    # transient allocations that a polling thread could miss.
    peak_rss = getattr(memory, "peak_wset", None)
    metrics = {
        "input_path": str(args.input_path.resolve()),
        "input_bytes": args.input_path.stat().st_size,
        "input_sha256": digest,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "merge_count": len(merges),
        "special_tokens": special_tokens,
        "num_workers": args.num_workers,
        "training_wall_seconds": elapsed,
        "training_cpu_seconds": cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system,
        "cpu_measurement": "Parent process CPU time only; excludes worker CPU time",
        "baseline_rss_bytes": baseline,
        "rss_after_training_bytes": memory.rss,
        "process_peak_rss_bytes": peak_rss,
        "process_tree_peak_rss_bytes": tree_peak_rss[0],
        "process_tree_sampling_interval_seconds": 0.05,
        "process_peak_commit_bytes": getattr(memory, "peak_pagefile", None),
        "memory_measurement": "Windows OS peak working set for this process through training completion",
        "timing_scope": "train_bpe only; includes reading/pretokenization/merging, excludes hashing/serialization",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(),
        "total_ram_bytes": psutil.virtual_memory().total,
        "longest_tokens": [],
        "longest_non_special_tokens": [],
    }
    special_bytes = {token.encode("utf-8") for token in special_tokens}
    for key, entries in (
        ("longest_tokens", list(vocab.items())),
        ("longest_non_special_tokens", [(i, token) for i, token in vocab.items() if token not in special_bytes]),
    ):
        max_length = max(map(lambda entry: len(entry[1]), entries), default=0)
        metrics[key] = [
            {
                "id": i,
                "byte_length": len(token),
                "hex": token.hex(),
                "text": token.decode("utf-8", errors="backslashreplace"),
            }
            for i, token in entries
            if len(token) == max_length
        ]

    vocab_path = args.output_dir / "vocab.json"
    merges_path = args.output_dir / "merges.json"
    vocab_path.write_text(json.dumps({str(i): token.hex() for i, token in vocab.items()}, indent=2), encoding="utf-8")
    merges_path.write_text(json.dumps([(a.hex(), b.hex()) for a, b in merges], indent=2), encoding="utf-8")
    with (args.output_dir / "vocab_readable.tsv").open("w", encoding="utf-8") as target:
        target.write("id\tbyte_length\thex\tutf8_text_json\n")
        for i, token in vocab.items():
            display = json.dumps(token.decode("utf-8", errors="backslashreplace"), ensure_ascii=False)
            target.write(f"{i}\t{len(token)}\t{token.hex()}\t{display}\n")

    # Check the disk artifacts, not just the in-memory representation.
    restored_vocab = {int(i): bytes.fromhex(token) for i, token in json.loads(vocab_path.read_text()).items()}
    restored_merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in json.loads(merges_path.read_text())]
    assert restored_vocab == vocab
    assert restored_merges == merges
    assert all(token in vocab.values() for token in special_bytes)
    metrics["serialization_roundtrip_verified"] = True
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
