# BPE training in C++, C#, Java, and Python

These are standalone trainers for the TinyStories experiment: valid UTF-8 input,
one reserved special token (`<|endoftext|>`, ID 256), and a configurable maximum
vocabulary size. They do not replace the Python assignment adapter or implement
inference-time encoding/decoding.

All implementations use the GPT-2 pretokenization expression, 32 MiB tentative
chunks aligned to document separators, frequency-weighted distinct pretokens,
an inverted pair-to-word index, and a lazy priority queue. Frequency ties select
the lexicographically greatest pair of **unsigned byte strings**, not the largest
token IDs. Counts are 64-bit in the three ports. Input newline normalization
matches Python's text reader.

## Build

Requirements: Python project environment, GCC/G++, CMake, Ninja, .NET 8 SDK,
and JDK 11 or newer. From the repository root:

```powershell
uv run python benchmarks/bpe_native/build.py
```

The build script downloads PCRE2 10.46 from its official GitHub release,
verifies the pinned SHA-256, and builds a static 8-bit library with Unicode
and JIT enabled. Downloaded sources, build products, and .NET intermediates
are ignored by Git. It discovers Java from PATH, JAVA_HOME, or ~/.jdks;
`--java-home` can explicitly select the JDK during building.

Sources:

- `cpp/main.cpp`: C++17 with PCRE2 JIT and `std::async` workers.
- `csharp/Program.cs`: .NET 8 compiled regex per worker, span-based matching,
  text-frequency counting before UTF-8 conversion, and task workers.
- `java/BpeBenchmark.java`: Java 11 regex with Unicode character classes and
  a fixed thread pool.
- `python_run.py`: the existing Python trainer with lightweight stage timers.

Each native executable accepts `INPUT OUTPUT_DIR VOCAB_SIZE WORKERS`.
It writes `vocab.tsv` (ID, hex bytes), `merges.tsv` (left/right hex bytes), and
`timings.json`. The byte strings are losslessly represented as hex.

## Verify and benchmark

```powershell
uv run python benchmarks/bpe_native/run_benchmarks.py --validate-only
uv run python benchmarks/bpe_native/run_benchmarks.py --runs 3 --workers 8
```

Validation compares the complete vocabulary including IDs and every merge in
order against Python on `corpus.en`, the 5 MB TinyStories fixture, and generated
Unicode/whitespace/overlap cases. Each full-corpus run is also compared against
the existing 10,000-token Python result when the dataset SHA-256 matches.

The benchmark runs trainers **sequentially**, each in a fresh process, and rotates
language order between repetitions. The report uses medians of three runs.
Python uses processes; the native versions use threads. All use eight workers
on the full corpus. There is no claim that these ports are equally optimized.

- Internal training time includes reading, pretokenization, setup, and merging;
  it excludes executable/runtime startup and output serialization.
- Process wall time includes startup, training, and serialization. Fresh JVM
  and .NET processes include their normal JIT/tiering behavior; these are not
  steady-state, prewarmed microbenchmarks.
- Memory is the maximum sampled sum of parent and descendant RSS, sampled
  every 20 ms. Samples can miss brief peaks and can count shared pages more
  than once. GC heap policy affects managed-runtime RSS.
- File caches are not cleared. Results describe this host and these specific
  implementations, including their regex engines, data structures, and runtime
  defaults; they do not establish a universal language ranking.
- Unicode versions and regex engines differ. Equality is checked on the actual
  dataset and supplied validation cases, not asserted for every Unicode input.
  In particular .NET regex category handling is based on UTF-16 code units,
  whereas Java/PCRE2 can recognize supplementary-plane letters as code points.

Detailed measurements are written to
`artifacts/bpe_language_benchmark/raw_results.json` and `summary.json`.
Large per-run vocabularies, merges, and logs are ignored by Git.

## C# investigation and corrected implementation

The original comparison used one shared Regex and per-occurrence Match objects
and byte conversions. That implementation was disproportionately slow. The
current default (`BPE_CSHARP_PRETOKEN_MODE=local`) gives each worker an exclusive
Regex, uses `EnumerateMatches` over document spans, and encodes only distinct
pretokens after text-frequency aggregation. Dictionary increments use one lookup.
The regex expression and BPE merge rules are unchanged.

The old paths remain available for controlled experiments through the
`BPE_CSHARP_PRETOKEN_MODE` environment variable: `legacy`, `span`, `text`, `local`.
`span` removes document/Match materialization; `text` additionally delays UTF-8
conversion and changes dictionary lookup; `local` additionally isolates Regex
instances per worker. GC configuration is independent of these code paths.

```powershell
uv run python benchmarks/bpe_native/investigate_csharp.py --runs 2
uv run python benchmarks/bpe_native/investigate_csharp.py --full --mode local --runs 3
uv run python benchmarks/bpe_native/investigate_csharp.py --full --mode local --server-gc --runs 3
```

Full-corpus medians after correction: 8.30 s with default workstation GC, or
7.58 s with server GC and eight heaps. The original 73.69 s result is retained
as historical data, not a result for the corrected implementation. The original
Java median was 15.90 s. GC mode does not by itself explain the original gap.
Full diagnostic measurements and caveats are in
`artifacts/bpe_csharp_investigation/REPORT.md`.

Regex references: [PCRE2 API](https://www.pcre2.org/doc/pcre2api/),
[Java Pattern](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/regex/Pattern.html),
[.NET character classes](https://learn.microsoft.com/en-us/dotnet/standard/base-types/character-classes-in-regular-expressions).
