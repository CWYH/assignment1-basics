"""Controlled C# pretokenization/GC ablations with exact-output validation."""

import argparse
import json
import os
import statistics

from run_benchmarks import ROOT, commands, read_artifacts, run_one

from cs336_basics.train_bpe import _document_ranges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--mode", choices=["legacy", "span", "text", "local"], default="local")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--server-gc", action="store_true")
    args = parser.parse_args()
    original = ROOT / "data/TinyStoriesV2-GPT4-train.txt"
    output = ROOT / "artifacts/bpe_csharp_investigation"
    output.mkdir(parents=True, exist_ok=True)
    trainers = commands()
    if args.full:
        input_path = original
        reference_path = ROOT / "artifacts/tinystories_bpe_10000_parallel"
        expected = (
            {int(i): bytes.fromhex(t) for i, t in json.loads((reference_path / "vocab.json").read_text()).items()},
            [
                tuple(bytes.fromhex(t) for t in pair)
                for pair in json.loads((reference_path / "merges.json").read_text())
            ],
        )
        variants = [(args.mode, args.server_gc)]
        label = f"full_{args.mode}_{'server' if args.server_gc else 'workstation'}"
    else:
        input_path = ROOT / "data/TinyStories_first_256MiB.txt"
        end = _document_ranges(original, b"<|endoftext|>", 32 * 1024**2)[7][1]
        if not input_path.exists() or input_path.stat().st_size != end:
            with original.open("rb") as source, input_path.open("wb") as target:
                remaining = end
                while remaining:
                    block = source.read(min(1024**2, remaining))
                    if not block:
                        raise RuntimeError("Unexpected EOF")
                    target.write(block)
                    remaining -= len(block)
        reference = output / "runs/sample_reference"
        run_one(trainers["Python"], input_path, reference, 10000, 8)
        expected = read_artifacts(reference)
        modes = ("local",) if args.local_only else ("legacy", "span", "text", "local")
        variants = [(mode, server) for mode in modes for server in (False, True)]
        label = "sample_local" if args.local_only else "sample_ablation"

    results = {}
    for repeat in range(args.runs):
        order = variants[repeat % len(variants) :] + variants[: repeat % len(variants)]
        for mode, server in order:
            name = f"{mode}_{'server' if server else 'workstation'}"
            env = os.environ.copy()
            env.update(BPE_CSHARP_PRETOKEN_MODE=mode, DOTNET_gcServer="1" if server else "0", DOTNET_GCHeapCount="8")
            print(f"{label}: run {repeat + 1}/{args.runs}, {name}", flush=True)
            metrics = run_one(
                trainers["C#"], input_path, output / "runs" / label / name / str(repeat + 1), 10000, 8, env=env
            )
            assert metrics["server_gc"] == server
            assert read_artifacts(output / "runs" / label / name / str(repeat + 1)) == expected
            metrics["byte_exact_match"] = True
            results.setdefault(name, []).append(metrics)
            print(json.dumps(metrics), flush=True)
            (output / f"{label}_raw.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    medians = {
        name: {
            key: statistics.median(run[key] for run in runs)
            for key in runs[0]
            if isinstance(runs[0][key], (int, float)) and not isinstance(runs[0][key], bool)
        }
        for name, runs in results.items()
    }
    summary = {
        "input_bytes": input_path.stat().st_size,
        "runs": args.runs,
        "workers": 8,
        "all_outputs_byte_exact": True,
        "medians": medians,
    }
    (output / f"{label}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
