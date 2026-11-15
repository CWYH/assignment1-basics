# C# BPE 性能问题的定位与修正

这次差距主要来自初版 C# 移植实现，不是 C# 必然比 Java 慢。
同样也不存在“C# 理论上一定比 Java 快”的通用结论。

## 主要问题：共享 Regex 的执行器缓存

初版的 8 个任务共享一个静态 Regex 实例。线程安全保证调用正确，
但不保证密集并行调用没有性能损失。.NET 8.0.11 的源码显示，每个 Regex
只有一个 `_runner` 缓存槽，匹配时通过 `Interlocked.Exchange` 获取执行器，
没有可用缓存就调用 `CreateRunner()`，结束后再写回缓存。
多个任务高频匹配时可能造成缓存竞争和重复创建执行器。

在其余逻辑不变的 `text` / `local` 对照中，仅改成每个工作单元独占 Regex，
约 256 MiB 样本上的预分词中位耗时从 4.48 秒降到 0.89 秒，累计托管分配量
从 19.41 GB 降到 2.91 GB。这强烈支持执行器缓存竞争是本实现的重要瓶颈。
这里没有通过调用栈采样逐项统计分配类型，因此不把全部分配差额都归因于某一个对象类型。

源码依据：[.NET 8.0.11 Regex.cs](https://github.com/dotnet/runtime/blob/v8.0.11/src/libraries/System.Text.RegularExpressions/src/System/Text/RegularExpressions/Regex.cs#L343)。

## 其他可避免的工作

- `Regex.Matches` 为匹配结果创建 Match 对象；改用 span 上的
  `Regex.EnumerateMatches`，只获取匹配位置和长度。
- `text.Split` 创建文档子串和数组；改为扫描特殊 token 的位置，处理对应的文档 span。
- 初版对每一次片段出现都做字符串 → UTF-8 数组 → Latin-1 字节容器转换；
  现在先按原始字符串统计次数，再为每个不同片段执行一次转换。
  这不改变 UTF-8 字节片段的加权频率。
- 文本计数器使用 `CollectionsMarshal.GetValueRefOrAddDefault`，避免一次增加计数做两次字典查找。

## 消融实验

样本取完整训练文件开头约 256 MiB，并对齐到特殊 token 边界：268,437,362 字节。
8 个工作单元，每项两次独立进程运行，下表为中位数，全部使用 Workstation GC。

| 版本 | 预分词耗时 | 预分词累计托管分配 | GC 暂停时间 |
|---|---:|---:|---:|
| 原实现：共享 Regex、Matches、每次出现都转换编码 | 8.62 秒 | 39.52 GB | 1.67 秒 |
| span：减少文档和 Match 对象 | 5.96 秒 | 29.40 GB | 0.42 秒 |
| text：另外推迟编码转换、减少字典查找 | 4.48 秒 | 19.41 GB | 0.31 秒 |
| local：另外为每个工作单元独立创建 Regex | **0.89 秒** | **2.91 GB** | **0.07 秒** |

分配总量按十进制 GB 表示，是运行中不断分配并回收的累计字节数，**不是峰值内存**。
`span` 和 `text` 各自包含一组相关改动，不将它们组内各项单独量化归因。

Server GC 也有作用，但不是主要修正：只将原样本实现切换到 Server GC，
预分词从 8.62 秒降到 6.46 秒，仍明显慢于 local。
Server GC 适合高吞吐多核工作负载，但更合适的配置仍需实测。
参见 [Microsoft GC 文档](https://learn.microsoft.com/en-us/dotnet/standard/garbage-collection/workstation-server-gc)。

## 完整 TinyStories 复核

输入仍为 2,227,753,162 字节，词表 10,000，8 个工作单元。
下面各项总耗时均为三轮中位数，包含读取、预分词、初始化和 BPE 合并，
不含进程启动与结果序列化，也没有启用 profiler。

| 版本 | 训练总耗时 |
|---|---:|
| 初版 C# / Workstation GC（原基准） | 73.69 秒 |
| 修正 C# / Workstation GC | **8.30 秒** |
| 修正 C# / Server GC、8 个 GC 堆 | **7.58 秒** |
| Java（原基准，未修改） | 15.90 秒 |
| C++（原基准，未修改） | 6.22 秒 |

所有 6 次完整训练均与原 Python 的词表 ID 和全部有序合并规则逐项一致，
初版及修正版也在英文参考样本、5 MB TinyStories 和 Unicode/空白边界样例上验证一致。
单次完整训练的预分词累计托管分配约 24.10 GB；默认 GC 下的进程 RSS 峰值三轮中位数
约 3.85 GiB，Server GC 配置下约 2.77 GiB（20 ms 采样）。因此分配总量显著降低
不等于进程 RSS 必然更小：GC 堆增长和回收策略也会影响峰值。

修正后的 C# 在本机、本实现上确实快于这份 Java 实现，但不意味着所有 C# 程序都比 Java 快。
历史 Java/C++ 数据没有在本次修正中重新计时；它们是此前同机、同数据的三轮结果。

原始数据见 `sample_ablation_raw.json`、`sample_local_raw.json`、
`full_local_workstation_raw.json` 和 `full_local_server_raw.json`；相应中位数见同名 `_summary.json`。
复现入口是 `benchmarks/bpe_native/investigate_csharp.py`。
