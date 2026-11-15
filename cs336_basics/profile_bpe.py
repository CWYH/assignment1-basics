"""Profile full-corpus training and one representative task per worker."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import time
from pathlib import Path

import cs336_basics.train_bpe as training


_original_count_chunk = training._count_chunk
_worker_profiled = False


def _profile_count_chunk(task):
    global _worker_profiled
    input_path, start, end, special_token, output_dir = task
    profiler = cProfile.Profile() if not _worker_profiled else None
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    if profiler is not None:
        profiler.enable()
    result = _original_count_chunk((input_path, start, end, special_token))
    if profiler is not None:
        profiler.disable()
    metrics = {
        "pid": os.getpid(),
        "start": start,
        "end": end,
        "wall_seconds": time.perf_counter() - wall_start,
        "cpu_seconds": time.process_time() - cpu_start,
        "profiled": profiler is not None,
    }
    output = Path(output_dir)
    (output / f"task_{start}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if profiler is not None:
        profiler.dump_stats(str(output / f"worker_{os.getpid()}.prof"))
        _worker_profiled = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timeline = {}
    original_count_pretokens = training._count_pretokens
    original_heappop = training.heappop

    def count_pretokens(*positional, **keywords):
        timeline["pretoken_start"] = time.perf_counter()
        result = original_count_pretokens(*positional, **keywords)
        timeline["pretoken_end"] = time.perf_counter()
        return result

    def heappop(heap):
        if "merge_start" not in timeline:
            timeline["merge_start"] = time.perf_counter()
        return original_heappop(heap)

    # Add an output directory to tasks without changing the training API.
    class ProfilePool(training.ProcessPoolExecutor):
        def map(self, function, tasks):
            return super().map(
                _profile_count_chunk,
                [(*task, str(args.output_dir.resolve())) for task in tasks],
            )

    training.ProcessPoolExecutor = ProfilePool
    training._count_pretokens = count_pretokens
    training.heappop = heappop
    profiler = cProfile.Profile()
    print("Profiling full-corpus training and the first task of each worker", flush=True)
    start = time.perf_counter()
    profiler.enable()
    vocab, merges = training.train_bpe(args.input_path, 10_000, ["<|endoftext|>"], num_workers=args.num_workers)
    profiler.disable()
    finish = time.perf_counter()
    profiler.dump_stats(str(args.output_dir / "parent.prof"))

    baseline = Path("artifacts/tinystories_bpe_10000_parallel")
    reference_vocab = {int(i): bytes.fromhex(t) for i, t in json.loads((baseline / "vocab.json").read_text()).items()}
    reference_merges = [
        tuple(bytes.fromhex(t) for t in pair) for pair in json.loads((baseline / "merges.json").read_text())
    ]
    assert vocab == reference_vocab
    assert merges == reference_merges
    tasks = [json.loads(path.read_text()) for path in args.output_dir.glob("task_*.json")]
    summary = {
        "total_wall_seconds_with_profiling": finish - start,
        "pretokenization_wall_seconds": timeline["pretoken_end"] - timeline["pretoken_start"],
        "pair_index_and_heap_setup_wall_seconds": timeline["merge_start"] - timeline["pretoken_end"],
        "merge_loop_wall_seconds": finish - timeline["merge_start"],
        "worker_task_cpu_seconds_sum": sum(task["cpu_seconds"] for task in tasks),
        "task_count": len(tasks),
        "profiled_task_count": sum(task["profiled"] for task in tasks),
        "matches_unprofiled_training": True,
        "note": "cProfile adds overhead. Each worker profiles only its first task; CPU sums are not wall time.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    worker_files = list(args.output_dir.glob("worker_*.prof"))
    for name, files in (("parent", [args.output_dir / "parent.prof"]), ("workers_sample", worker_files)):
        stream = io.StringIO()
        stats = pstats.Stats(str(files[0]), stream=stream)
        for path in files[1:]:
            stats.add(str(path))
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        stats.sort_stats("tottime").print_stats(30)
        (args.output_dir / f"{name}.txt").write_text(stream.getvalue(), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
