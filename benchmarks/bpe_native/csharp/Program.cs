using System.Diagnostics;
using System.Runtime;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

static class BpeBenchmark
{
    const string Special = "<|endoftext|>";
    static readonly Regex Pretokens = new(
        @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);
    static readonly UTF8Encoding Utf8 = new(false, true);
    // Preserve the initial implementation for controlled A/B measurements.
    static readonly string PretokenMode = Environment.GetEnvironmentVariable("BPE_CSHARP_PRETOKEN_MODE") ?? "local";
    readonly record struct Entry(long Count, ulong Pair);
    static ulong Pack(int a, int b) => ((ulong)(uint)a << 32) | (uint)b;
    static int First(ulong pair) => (int)(pair >> 32);
    static int Second(ulong pair) => (int)(uint)pair;
    static void Add<TKey>(Dictionary<TKey, long> counts, TKey key, long value) where TKey : notnull
    {
        counts.TryGetValue(key, out long previous);
        counts[key] = previous + value;
    }
    static List<long> Boundaries(string path)
    {
        long size = new FileInfo(path).Length;
        var result = new List<long> { 0 };
        byte[] separator = Utf8.GetBytes(Special);
        byte[] block = new byte[65536];
        using var file = File.OpenRead(path);
        for (long at = 32L * 1024 * 1024; at < size; at += 32L * 1024 * 1024)
        {
            if (at <= result[^1]) continue;
            long offset = at;
            bool found = false;
            while (offset < size)
            {
                file.Position = offset;
                int length = (int)Math.Min(block.Length, size - offset);
                file.ReadExactly(block.AsSpan(0, length));
                int index = block.AsSpan(0, length).IndexOf(separator);
                if (index >= 0) { result.Add(offset + index); found = true; break; }
                if (offset + length == size) break;
                offset += length - separator.Length + 1;
            }
            if (!found) break;
        }
        result.Add(size);
        return result;
    }
    static Dictionary<string, long> Pretokenize(string path, int workers)
    {
        var offsets = Boundaries(path);
        int next = -1;
        var jobs = Enumerable.Range(0, Math.Min(workers, offsets.Count - 1)).Select(_ => Task.Run(() =>
        {
            var counts = new Dictionary<string, long>(StringComparer.Ordinal);
            // Regex is thread-safe, but .NET 8 keeps only one cached runner per
            // instance. Exclusive worker instances avoid hot-path cache churn.
            var regex = PretokenMode == "local" ? new Regex(Pretokens.ToString(), Pretokens.Options) : Pretokens;
            using var file = File.OpenRead(path);
            while (true)
            {
                int task = Interlocked.Increment(ref next);
                if (task + 1 >= offsets.Count) break;
                file.Position = offsets[task];
                byte[] raw = new byte[checked((int)(offsets[task + 1] - offsets[task]))];
                file.ReadExactly(raw);
                string text = Utf8.GetString(raw).Replace("\r\n", "\n").Replace('\r', '\n');
                if (PretokenMode == "legacy")
                {
                    foreach (string segment in text.Split(Special, StringSplitOptions.None))
                        foreach (Match match in Pretokens.Matches(segment))
                            Add(counts, Encoding.Latin1.GetString(Utf8.GetBytes(match.Value)), 1);
                }
                else
                {
                    int begin = 0;
                    while (begin < text.Length)
                    {
                        int end = text.IndexOf(Special, begin, StringComparison.Ordinal);
                        if (end < 0) end = text.Length;
                        CountSegment(text.AsSpan(begin, end - begin), counts, regex);
                        if (end == text.Length) break;
                        begin = end + Special.Length;
                    }
                }
            }
            return counts;
        })).ToArray();
        var combined = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var job in jobs)
            foreach (var item in job.GetAwaiter().GetResult()) Add(combined, item.Key, item.Value);
        if (PretokenMode is not ("text" or "local")) return combined;
        // UTF-8 is deterministic: counting text first and encoding each distinct
        // pretoken once gives exactly the same weighted byte sequences.
        var bytes = new Dictionary<string, long>(combined.Count, StringComparer.Ordinal);
        foreach (var item in combined)
            Add(bytes, Encoding.Latin1.GetString(Utf8.GetBytes(item.Key)), item.Value);
        return bytes;
    }
    static void CountSegment(ReadOnlySpan<char> segment, Dictionary<string, long> counts, Regex regex)
    {
        foreach (var match in regex.EnumerateMatches(segment))
        {
            string key = segment.Slice(match.Index, match.Length).ToString();
            if (PretokenMode == "span")
            {
                Add(counts, Encoding.Latin1.GetString(Utf8.GetBytes(key)), 1);
            }
            else
            {
                ref long count = ref CollectionsMarshal.GetValueRefOrAddDefault(counts, key, out _);
                count++;
            }
        }
    }
    static Dictionary<ulong, long> Pairs(int[] word)
    {
        var counts = new Dictionary<ulong, long>();
        for (int i = 1; i < word.Length; i++) Add(counts, Pack(word[i - 1], word[i]), 1);
        return counts;
    }
    static string Hex(string bytes) => Convert.ToHexString(Encoding.Latin1.GetBytes(bytes)).ToLowerInvariant();
    static int Main(string[] args)
    {
        try
        {
            if (args.Length != 4) throw new ArgumentException("Usage: BpeBenchmark INPUT OUTPUT_DIR VOCAB_SIZE WORKERS");
            if (PretokenMode is not ("legacy" or "span" or "text" or "local"))
                throw new ArgumentException("BPE_CSHARP_PRETOKEN_MODE must be legacy, span, text, or local");
            int size = int.Parse(args[2]), workers = int.Parse(args[3]);
            if (size < 257 || workers < 1) throw new ArgumentException("Need vocab >=257 and workers >=1");
            Directory.CreateDirectory(args[1]);
            var clock = Stopwatch.StartNew();
            long allocatedBefore = GC.GetTotalAllocatedBytes(true);
            TimeSpan pausedBefore = GC.GetTotalPauseDuration();
            int[] collectionsBefore = Enumerable.Range(0, 3).Select(GC.CollectionCount).ToArray();
            var counts = Pretokenize(args[0], workers);
            double counted = clock.Elapsed.TotalSeconds;
            long pretokenAllocated = GC.GetTotalAllocatedBytes(true) - allocatedBefore;
            double pretokenGcPause = (GC.GetTotalPauseDuration() - pausedBefore).TotalSeconds;
            int[] pretokenCollections = Enumerable.Range(0, 3).Select(i => GC.CollectionCount(i) - collectionsBefore[i]).ToArray();
            var vocab = Enumerable.Range(0, 256).Select(i => ((char)i).ToString()).ToList();
            vocab.Add(Special);
            var words = new List<int[]>();
            var frequencies = new List<long>();
            var global = new Dictionary<ulong, long>();
            var index = new Dictionary<ulong, HashSet<int>>();
            foreach (var item in counts)
            {
                int[] word = item.Key.Select(c => (int)c).ToArray();
                foreach (var pair in Pairs(word))
                {
                    Add(global, pair.Key, pair.Value * item.Value);
                    if (!index.TryGetValue(pair.Key, out var ids)) index[pair.Key] = ids = new();
                    ids.Add(words.Count);
                }
                words.Add(word); frequencies.Add(item.Value);
            }
            var comparer = Comparer<Entry>.Create((a, b) =>
            {
                int cmp = b.Count.CompareTo(a.Count);
                if (cmp != 0) return cmp;
                cmp = string.CompareOrdinal(vocab[First(b.Pair)], vocab[First(a.Pair)]);
                return cmp != 0 ? cmp : string.CompareOrdinal(vocab[Second(b.Pair)], vocab[Second(a.Pair)]);
            });
            var heap = new PriorityQueue<Entry, Entry>(comparer);
            foreach (var pair in global)
            {
                var entry = new Entry(pair.Value, pair.Key); heap.Enqueue(entry, entry);
            }
            double initialized = clock.Elapsed.TotalSeconds;
            var merges = new List<ulong>();
            while (vocab.Count < size && heap.Count > 0)
            {
                Entry best = heap.Dequeue();
                if (!global.TryGetValue(best.Pair, out long current) || current != best.Count) continue;
                int token = vocab.Count;
                vocab.Add(vocab[First(best.Pair)] + vocab[Second(best.Pair)]); merges.Add(best.Pair);
                var changed = new HashSet<ulong>();
                foreach (int id in index[best.Pair].ToArray())
                {
                    var old = Pairs(words[id]);
                    var updated = new List<int>();
                    var word = words[id];
                    for (int i = 0; i < word.Length;)
                    {
                        if (i + 1 < word.Length && Pack(word[i], word[i + 1]) == best.Pair) { updated.Add(token); i += 2; }
                        else updated.Add(word[i++]);
                    }
                    var now = Pairs(updated.ToArray());
                    foreach (ulong pair in old.Keys.Union(now.Keys))
                    {
                        old.TryGetValue(pair, out long before); now.TryGetValue(pair, out long after);
                        long delta = after - before;
                        if (delta != 0) { Add(global, pair, delta * frequencies[id]); changed.Add(pair); }
                        if (after == 0) index[pair].Remove(id);
                        else if (before == 0)
                        {
                            if (!index.TryGetValue(pair, out var ids)) index[pair] = ids = new();
                            ids.Add(id);
                        }
                    }
                    words[id] = updated.ToArray();
                }
                foreach (ulong pair in changed)
                {
                    if (global[pair] > 0)
                    {
                        var entry = new Entry(global[pair], pair); heap.Enqueue(entry, entry);
                    }
                    else { global.Remove(pair); index.Remove(pair); }
                }
            }
            double finished = clock.Elapsed.TotalSeconds;
            using (var target = new StreamWriter(Path.Combine(args[1], "vocab.tsv"), false, new UTF8Encoding(false)))
                for (int i = 0; i < vocab.Count; i++) target.WriteLine($"{i}\t{Hex(vocab[i])}");
            using (var target = new StreamWriter(Path.Combine(args[1], "merges.tsv"), false, new UTF8Encoding(false)))
                foreach (ulong pair in merges) target.WriteLine($"{Hex(vocab[First(pair)])}\t{Hex(vocab[Second(pair)])}");
            string metrics = JsonSerializer.Serialize(new
            {
                pretoken_seconds = counted,
                setup_seconds = initialized - counted,
                merge_seconds = finished - initialized,
                training_seconds = finished,
                unique_pretokens = counts.Count,
                pretoken_mode = PretokenMode,
                server_gc = GCSettings.IsServerGC,
                pretoken_allocated_bytes = pretokenAllocated,
                pretoken_gc_pause_seconds = pretokenGcPause,
                pretoken_gc_collections = pretokenCollections,
            });
            File.WriteAllText(Path.Combine(args[1], "timings.json"), metrics);
            Console.WriteLine(metrics);
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}
