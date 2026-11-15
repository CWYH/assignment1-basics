"""Run each trainer in a fresh process, check byte-exact output, and report medians."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def find_java():
    if java := shutil.which("java"):
        return java
    if java_home := os.environ.get("JAVA_HOME"):
        return str(Path(java_home) / "bin" / "java.exe")
    for folder in sorted((Path.home() / ".jdks").glob("*")):
        if (folder / "bin" / "java.exe").exists():
            return str(folder / "bin" / "java.exe")
    raise RuntimeError("Set JAVA_HOME or add java to PATH")


def commands():
    return {
        "Python": [sys.executable, str(HERE / "python_run.py")],
        "C++": [str(HERE / "build" / "cpp" / ("bpe_cpp.exe" if os.name == "nt" else "bpe_cpp"))],
        "C#": ["dotnet", str(HERE / "csharp" / "bin" / "Release" / "net8.0" / "BpeBenchmark.dll")],
        # Bound the heap so that ergonomic defaults do not consume most host RAM.
        "Java": [find_java(), "-Xms64m", "-Xmx2g", "-cp", str(HERE / "build" / "java"), "BpeBenchmark"],
    }


def read_artifacts(output):
    vocab = {}
    for line in (output / "vocab.tsv").read_text(encoding="utf-8-sig").splitlines():
        i, token = line.split("\t")
        vocab[int(i)] = bytes.fromhex(token)
    merges = []
    for line in (output / "merges.tsv").read_text(encoding="utf-8-sig").splitlines():
        a, b = line.split("\t")
        merges.append((bytes.fromhex(a), bytes.fromhex(b)))
    assert all(vocab[i] == bytes([i]) for i in range(256))
    assert vocab[256] == b"<|endoftext|>"
    assert len(vocab) == 257 + len(merges)
    for i, (a, b) in enumerate(merges, 257):
        assert vocab[i] == a + b
    return vocab, merges


def run_one(command, input_path, output, size, workers, env=None):
    output.mkdir(parents=True, exist_ok=True)
    peak = 0
    started = time.perf_counter()
    with (output / "stdout.txt").open("w", encoding="utf-8") as stdout:
        with (output / "stderr.txt").open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                [*command, str(input_path), str(output), str(size), str(workers)],
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                env=env,
            )
            observed = psutil.Process(process.pid)
            while process.poll() is None:
                rss = 0
                try:
                    members = [observed, *observed.children(recursive=True)]
                except psutil.NoSuchProcess:
                    members = []
                for member in members:
                    try:
                        rss += member.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak = max(peak, rss)
                time.sleep(0.02)
    if process.returncode:
        raise RuntimeError(f"Command failed: {command}\n{(output / 'stderr.txt').read_text(encoding='utf-8')}")
    metrics = json.loads((output / "timings.json").read_text(encoding="utf-8-sig"))
    metrics["process_wall_seconds"] = time.perf_counter() - started
    metrics["sampled_process_tree_peak_rss_bytes"] = peak
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/bpe_language_benchmark")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trainers = commands()

    # Each implementation must match on actual fixtures and boundary cases
    # before expensive full-corpus benchmarking.
    edge_path = output / "edge_cases.txt"
    edge_path.write_bytes(
        (
            "aaa aaaa abab can't we're!\r\n你好 café 😀\t  \n<|endoftext|><|endoftext|>hello  hi\rbye\u0085\u00a0end"
        ).encode()
    )
    for case, path, size in (
        ("corpus", ROOT / "tests/fixtures/corpus.en", 500),
        ("special_tokens", ROOT / "tests/fixtures/tinystories_sample_5M.txt", 1000),
        ("edge_cases", edge_path, 320),
    ):
        reference = None
        for language, command in trainers.items():
            case_output = output / "validation" / case / language.replace("+", "p").replace("#", "sharp")
            run_one(command, path, case_output, size, args.workers)
            actual = read_artifacts(case_output)
            if reference is None:
                reference = actual
            else:
                assert actual == reference, f"{language}: byte-exact mismatch on {case}"
        print(f"Validated all four languages: {case}", flush=True)
    if args.validate_only:
        return

    with args.input.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    previous = ROOT / "artifacts/tinystories_bpe_10000_parallel"
    expected = None
    if digest == "6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443":
        expected = (
            {int(i): bytes.fromhex(t) for i, t in json.loads((previous / "vocab.json").read_text()).items()},
            [tuple(bytes.fromhex(t) for t in pair) for pair in json.loads((previous / "merges.json").read_text())],
        )
    results = {language: [] for language in trainers}
    # Rotate order between repetitions; no concurrent timing of different languages.
    languages = list(trainers)
    for repetition in range(args.runs):
        order = languages[repetition % 4 :] + languages[: repetition % 4]
        for language in order:
            print(f"Run {repetition + 1}/{args.runs}: {language}", flush=True)
            run_output = output / f"run_{repetition + 1}" / language.replace("+", "p").replace("#", "sharp")
            metrics = run_one(trainers[language], args.input.resolve(), run_output, 10_000, args.workers)
            actual = read_artifacts(run_output)
            if expected is None:
                expected = actual
            assert actual == expected, f"Full-corpus mismatch: {language}, run {repetition + 1}"
            metrics["byte_exact_match"] = True
            results[language].append(metrics)
            print(json.dumps({"language": language, **metrics}), flush=True)
            (output / "raw_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    medians = {
        language: {
            key: statistics.median(run[key] for run in runs)
            for key in runs[0]
            if isinstance(runs[0][key], (int, float)) and not isinstance(runs[0][key], bool)
        }
        for language, runs in results.items()
    }
    summary = {
        "input_sha256": digest,
        "input_bytes": args.input.stat().st_size,
        "vocab_size": 10000,
        "special_token": "<|endoftext|>",
        "workers": args.workers,
        "repetitions": args.runs,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "logical_cpus": psutil.cpu_count(),
        "total_ram_bytes": psutil.virtual_memory().total,
        "memory_measurement": "20 ms samples of summed parent/child RSS; shared pages may be counted more than once",
        "timing": "Fresh processes; training timer excludes startup and serialization; process wall includes both",
        "all_outputs_byte_exact": True,
        "medians": medians,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
