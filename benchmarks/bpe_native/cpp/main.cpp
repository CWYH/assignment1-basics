#include <pcre2.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Counts = std::unordered_map<std::string, int64_t>;
using Pair = uint64_t;
using PairCounts = std::unordered_map<Pair, int64_t>;
using Clock = std::chrono::steady_clock;
const std::string special = "<|endoftext|>";
const char* pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
Pair pack(int a, int b) { return (uint64_t(uint32_t(a)) << 32) | uint32_t(b); }
int first(Pair p) { return int(p >> 32); }
int second(Pair p) { return int(uint32_t(p)); }
double seconds(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b-a).count(); }

std::vector<uint64_t> boundaries(const std::string& path) {
    uint64_t size = std::filesystem::file_size(path);
    std::vector<uint64_t> result{0};
    std::ifstream file(path, std::ios::binary);
    for (uint64_t at = 32ull*1024*1024; at < size; at += 32ull*1024*1024) {
        if (at <= result.back()) continue;
        uint64_t offset = at;
        bool found = false;
        while (offset < size) {
            file.clear(); file.seekg(offset);
            std::string block(size_t(std::min<uint64_t>(65536, size-offset)), '\0');
            file.read(block.data(), block.size());
            auto pos = block.find(special);
            if (pos != std::string::npos) { result.push_back(offset+pos); found = true; break; }
            if (offset+block.size() == size) break;
            offset += block.size() - special.size() + 1;
        }
        if (!found) break;
    }
    result.push_back(size);
    return result;
}

std::string normalize_newlines(const std::string& raw) {
    std::string text; text.reserve(raw.size());
    for (size_t i=0; i<raw.size(); ++i) {
        if (raw[i]=='\r') { text += '\n'; if (i+1<raw.size() && raw[i+1]=='\n') ++i; }
        else text += raw[i];
    }
    return text;
}

Counts pretokenize(const std::string& path, int workers) {
    auto offsets = boundaries(path);
    std::atomic<size_t> next{0};
    std::vector<std::future<Counts>> jobs;
    for (int worker=0; worker<std::min<int>(workers, offsets.size()-1); ++worker) {
        jobs.push_back(std::async(std::launch::async, [&] {
            int error; PCRE2_SIZE error_offset;
            auto code = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern), PCRE2_ZERO_TERMINATED,
                                      PCRE2_UTF | PCRE2_UCP, &error, &error_offset, nullptr);
            if (!code) throw std::runtime_error("Regex compilation failed");
            if (pcre2_jit_compile(code, PCRE2_JIT_COMPLETE) != 0) throw std::runtime_error("Regex JIT unavailable");
            auto match = pcre2_match_data_create_from_pattern(code, nullptr);
            Counts counts;
            std::ifstream file(path, std::ios::binary);
            while (true) {
                size_t task = next.fetch_add(1);
                if (task+1 >= offsets.size()) break;
                file.clear(); file.seekg(offsets[task]);
                std::string raw(size_t(offsets[task+1]-offsets[task]), '\0');
                file.read(raw.data(), raw.size());
                if (!file) throw std::runtime_error("Short input read");
                auto text = normalize_newlines(raw);
                size_t begin=0;
                while (begin<text.size()) {
                    auto end = text.find(special, begin);
                    if (end==std::string::npos) end=text.size();
                    PCRE2_SIZE offset=0;
                    while (offset<end-begin) {
                        int rc = pcre2_match(code, reinterpret_cast<PCRE2_SPTR>(text.data()+begin),
                                             end-begin, offset, PCRE2_NO_UTF_CHECK, match, nullptr);
                        if (rc==PCRE2_ERROR_NOMATCH) break;
                        if (rc<0) throw std::runtime_error("Regex matching failed: "+std::to_string(rc));
                        auto span = pcre2_get_ovector_pointer(match);
                        ++counts[text.substr(begin+span[0], span[1]-span[0])];
                        offset=span[1];
                    }
                    if (end==text.size()) break;
                    begin=end+special.size();
                }
            }
            pcre2_match_data_free(match); pcre2_code_free(code);
            return counts;
        }));
    }
    Counts counts;
    for (auto& job: jobs) for (const auto& item: job.get()) counts[item.first] += item.second;
    return counts;
}

PairCounts pairs(const std::vector<int>& word) {
    PairCounts result;
    for (size_t i=1; i<word.size(); ++i) ++result[pack(word[i-1], word[i])];
    return result;
}
struct Entry { int64_t count; Pair pair; };
struct Compare {
    const std::vector<std::string>* vocab;
    bool operator()(const Entry& a, const Entry& b) const {
        if (a.count!=b.count) return a.count<b.count;
        const auto& v=*vocab;
        int cmp=v[first(a.pair)].compare(v[first(b.pair)]);
        if (cmp) return cmp<0;
        return v[second(a.pair)]<v[second(b.pair)];
    }
};
std::string hex(const std::string& bytes) {
    const char* digits="0123456789abcdef";
    std::string out; out.reserve(2*bytes.size());
    for (unsigned char c: bytes) { out+=digits[c>>4]; out+=digits[c&15]; }
    return out;
}
int main(int argc, char** argv) {
    try {
        if (argc!=5) throw std::runtime_error("Usage: bpe_cpp INPUT OUTPUT_DIR VOCAB_SIZE WORKERS");
        int size=std::stoi(argv[3]), workers=std::stoi(argv[4]);
        if (size<257 || workers<1) throw std::runtime_error("Need vocab >=257 and workers >=1");
        std::filesystem::create_directories(argv[2]);
        auto start=Clock::now();
        auto counts=pretokenize(argv[1], workers);
        auto counted=Clock::now();
        std::vector<std::string> vocab;
        for (int i=0; i<256; ++i) vocab.emplace_back(1,char(i));
        vocab.push_back(special);
        std::vector<std::vector<int>> words;
        std::vector<int64_t> frequencies;
        PairCounts global;
        std::unordered_map<Pair, std::unordered_set<int>> index;
        for (const auto& item: counts) {
            std::vector<int> word;
            for (unsigned char c: item.first) word.push_back(c);
            int id=int(words.size());
            for (auto entry: pairs(word)) { global[entry.first]+=entry.second*item.second; index[entry.first].insert(id); }
            words.push_back(std::move(word)); frequencies.push_back(item.second);
        }
        std::priority_queue<Entry,std::vector<Entry>,Compare> heap(Compare{&vocab});
        for (auto item: global) heap.push({item.second,item.first});
        auto initialized=Clock::now();
        std::vector<Pair> merges;
        while (int(vocab.size())<size && !heap.empty()) {
            auto best=heap.top(); heap.pop();
            auto current=global.find(best.pair);
            if (current==global.end() || current->second!=best.count) continue;
            int token=int(vocab.size());
            vocab.push_back(vocab[first(best.pair)]+vocab[second(best.pair)]); merges.push_back(best.pair);
            std::vector<int> affected(index[best.pair].begin(), index[best.pair].end());
            std::unordered_set<Pair> changed;
            for (int id: affected) {
                auto old=pairs(words[id]);
                std::vector<int> updated;
                const auto& word=words[id];
                for (size_t i=0; i<word.size();) {
                    if (i+1<word.size() && pack(word[i],word[i+1])==best.pair) { updated.push_back(token); i+=2; }
                    else updated.push_back(word[i++]);
                }
                auto now=pairs(updated);
                std::unordered_set<Pair> all;
                for (auto item: old) all.insert(item.first);
                for (auto item: now) all.insert(item.first);
                for (Pair p: all) {
                    int64_t delta=now[p]-old[p];
                    if (delta) { global[p]+=delta*frequencies[id]; changed.insert(p); }
                    if (!now[p]) index[p].erase(id);
                    else if (!old[p]) index[p].insert(id);
                }
                words[id]=std::move(updated);
            }
            for (Pair p: changed) {
                if (global[p]>0) heap.push({global[p],p});
                else { global.erase(p); index.erase(p); }
            }
        }
        auto finish=Clock::now();
        std::ofstream vf(std::filesystem::path(argv[2])/"vocab.tsv");
        for (size_t i=0; i<vocab.size(); ++i) vf<<i<<'\t'<<hex(vocab[i])<<'\n';
        std::ofstream mf(std::filesystem::path(argv[2])/"merges.tsv");
        for (Pair p: merges) mf<<hex(vocab[first(p)])<<'\t'<<hex(vocab[second(p)])<<'\n';
        std::string metrics="{\"pretoken_seconds\":"+std::to_string(seconds(start,counted))+
            ",\"setup_seconds\":"+std::to_string(seconds(counted,initialized))+
            ",\"merge_seconds\":"+std::to_string(seconds(initialized,finish))+
            ",\"training_seconds\":"+std::to_string(seconds(start,finish))+
            ",\"unique_pretokens\":"+std::to_string(counts.size())+"}";
        std::ofstream(std::filesystem::path(argv[2])/"timings.json")<<metrics;
        std::cout<<metrics<<std::endl;
    } catch (const std::exception& error) { std::cerr<<error.what()<<std::endl; return 1; }
}
