import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.*;

public final class BpeBenchmark {
    static final String SPECIAL = "<|endoftext|>";
    static final Pattern PRETOKENS = Pattern.compile(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
        Pattern.UNICODE_CHARACTER_CLASS);
    static long pack(int a, int b) { return ((long)a << 32) | (b & 0xffffffffL); }
    static int first(long pair) { return (int)(pair >>> 32); }
    static int second(long pair) { return (int)pair; }
    static <K> void add(Map<K, Long> counts, K key, long value) { counts.merge(key, value, Long::sum); }
    static double elapsed(long start) { return (System.nanoTime() - start) / 1e9; }
    static List<Long> boundaries(String path) throws IOException {
        long size = Files.size(Path.of(path));
        List<Long> result = new ArrayList<>(); result.add(0L);
        byte[] separator = SPECIAL.getBytes(StandardCharsets.UTF_8);
        byte[] block = new byte[65536];
        try (RandomAccessFile file = new RandomAccessFile(path, "r")) {
            for (long at = 32L*1024*1024; at < size; at += 32L*1024*1024) {
                if (at <= result.get(result.size()-1)) continue;
                long offset = at;
                boolean found = false;
                while (offset < size) {
                    file.seek(offset);
                    int length = (int)Math.min(block.length, size-offset);
                    file.readFully(block, 0, length);
                    int match = -1;
                    outer: for (int i=0; i+separator.length<=length; i++) {
                        for (int j=0; j<separator.length; j++) if (block[i+j]!=separator[j]) continue outer;
                        match=i; break;
                    }
                    if (match>=0) { result.add(offset+match); found=true; break; }
                    if (offset+length==size) break;
                    offset += length-separator.length+1;
                }
                if (!found) break;
            }
        }
        result.add(size);
        return result;
    }
    static Map<String, Long> pretokenize(String path, int workers) throws Exception {
        List<Long> offsets = boundaries(path);
        AtomicInteger next = new AtomicInteger(0);
        int active = Math.min(workers, offsets.size()-1);
        ExecutorService pool = Executors.newFixedThreadPool(active);
        List<Future<Map<String, Long>>> jobs = new ArrayList<>();
        try {
            for (int worker=0; worker<active; worker++) jobs.add(pool.submit(() -> {
                Map<String, Long> counts = new HashMap<>();
                try (RandomAccessFile file = new RandomAccessFile(path, "r")) {
                    while (true) {
                        int task=next.getAndIncrement();
                        if (task+1>=offsets.size()) break;
                        file.seek(offsets.get(task));
                        byte[] raw = new byte[Math.toIntExact(offsets.get(task+1)-offsets.get(task))];
                        file.readFully(raw);
                        String text=new String(raw, StandardCharsets.UTF_8).replace("\r\n", "\n").replace('\r', '\n');
                        int begin=0;
                        while (begin<text.length()) {
                            int end=text.indexOf(SPECIAL, begin);
                            if (end<0) end=text.length();
                            // Opaque regions keep lookahead within this document.
                            Matcher matcher=PRETOKENS.matcher(text).region(begin, end);
                            while (matcher.find()) {
                                String bytes=new String(matcher.group().getBytes(StandardCharsets.UTF_8), StandardCharsets.ISO_8859_1);
                                add(counts, bytes, 1);
                            }
                            if (end==text.length()) break;
                            begin=end+SPECIAL.length();
                        }
                    }
                }
                return counts;
            }));
            Map<String, Long> combined=new HashMap<>();
            for (Future<Map<String, Long>> job: jobs)
                for (Map.Entry<String, Long> item: job.get().entrySet()) add(combined, item.getKey(), item.getValue());
            return combined;
        } finally { pool.shutdownNow(); }
    }
    static Map<Long, Long> pairs(int[] word) {
        Map<Long, Long> result=new HashMap<>();
        for (int i=1; i<word.length; i++) add(result, pack(word[i-1], word[i]), 1);
        return result;
    }
    static final class Entry {
        final long count, pair;
        Entry(long count, long pair) { this.count=count; this.pair=pair; }
    }
    static String hex(String bytes) {
        char[] digits="0123456789abcdef".toCharArray();
        StringBuilder out=new StringBuilder(bytes.length()*2);
        for (int i=0; i<bytes.length(); i++) {
            int b=bytes.charAt(i); out.append(digits[b>>>4]); out.append(digits[b&15]);
        }
        return out.toString();
    }
    public static void main(String[] args) throws Exception {
        if (args.length!=4) throw new IllegalArgumentException("Usage: BpeBenchmark INPUT OUTPUT_DIR VOCAB_SIZE WORKERS");
        int size=Integer.parseInt(args[2]), workers=Integer.parseInt(args[3]);
        if (size<257 || workers<1) throw new IllegalArgumentException("Need vocab >=257 and workers >=1");
        Path output=Path.of(args[1]); Files.createDirectories(output);
        long start=System.nanoTime();
        Map<String, Long> counts=pretokenize(args[0], workers);
        double counted=elapsed(start);
        List<String> vocab=new ArrayList<>();
        for (int i=0; i<256; i++) vocab.add(String.valueOf((char)i));
        vocab.add(SPECIAL);
        List<int[]> words=new ArrayList<>();
        List<Long> frequencies=new ArrayList<>();
        Map<Long, Long> global=new HashMap<>();
        Map<Long, Set<Integer>> index=new HashMap<>();
        for (Map.Entry<String, Long> item: counts.entrySet()) {
            int[] word=item.getKey().chars().toArray();
            int id=words.size();
            for (Map.Entry<Long, Long> entry: pairs(word).entrySet()) {
                add(global, entry.getKey(), entry.getValue()*item.getValue());
                index.computeIfAbsent(entry.getKey(), ignored -> new HashSet<>()).add(id);
            }
            words.add(word); frequencies.add(item.getValue());
        }
        PriorityQueue<Entry> heap=new PriorityQueue<>((a,b) -> {
            int cmp=Long.compare(b.count, a.count);
            if (cmp!=0) return cmp;
            cmp=vocab.get(first(b.pair)).compareTo(vocab.get(first(a.pair)));
            return cmp!=0 ? cmp : vocab.get(second(b.pair)).compareTo(vocab.get(second(a.pair)));
        });
        for (Map.Entry<Long, Long> item: global.entrySet()) heap.add(new Entry(item.getValue(), item.getKey()));
        double initialized=elapsed(start);
        List<Long> merges=new ArrayList<>();
        while (vocab.size()<size && !heap.isEmpty()) {
            Entry best=heap.remove();
            if (!global.containsKey(best.pair) || global.get(best.pair)!=best.count) continue;
            int token=vocab.size();
            vocab.add(vocab.get(first(best.pair))+vocab.get(second(best.pair))); merges.add(best.pair);
            Set<Long> changed=new HashSet<>();
            for (int id: new ArrayList<>(index.get(best.pair))) {
                int[] word=words.get(id);
                Map<Long, Long> old=pairs(word);
                int[] updated=new int[word.length];
                int length=0;
                for (int i=0; i<word.length;) {
                    if (i+1<word.length && pack(word[i],word[i+1])==best.pair) { updated[length++]=token; i+=2; }
                    else updated[length++]=word[i++];
                }
                updated=Arrays.copyOf(updated, length);
                Map<Long, Long> now=pairs(updated);
                Set<Long> all=new HashSet<>(old.keySet()); all.addAll(now.keySet());
                for (long pair: all) {
                    long before=old.getOrDefault(pair,0L), after=now.getOrDefault(pair,0L);
                    long delta=after-before;
                    if (delta!=0) { add(global, pair, delta*frequencies.get(id)); changed.add(pair); }
                    if (after==0) index.get(pair).remove(id);
                    else if (before==0) index.computeIfAbsent(pair, ignored -> new HashSet<>()).add(id);
                }
                words.set(id,updated);
            }
            for (long pair: changed) {
                if (global.get(pair)>0) heap.add(new Entry(global.get(pair),pair));
                else { global.remove(pair); index.remove(pair); }
            }
        }
        double finished=elapsed(start);
        try (BufferedWriter target=Files.newBufferedWriter(output.resolve("vocab.tsv"))) {
            for (int i=0; i<vocab.size(); i++) target.write(i+"\t"+hex(vocab.get(i))+"\n");
        }
        try (BufferedWriter target=Files.newBufferedWriter(output.resolve("merges.tsv"))) {
            for (long pair: merges) target.write(hex(vocab.get(first(pair)))+"\t"+hex(vocab.get(second(pair)))+"\n");
        }
        String metrics=String.format(Locale.ROOT,
            "{\"pretoken_seconds\":%.9f,\"setup_seconds\":%.9f,\"merge_seconds\":%.9f,\"training_seconds\":%.9f,\"unique_pretokens\":%d}",
            counted,initialized-counted,finished-initialized,finished,counts.size());
        Files.writeString(output.resolve("timings.json"),metrics);
        System.out.println(metrics);
    }
}
