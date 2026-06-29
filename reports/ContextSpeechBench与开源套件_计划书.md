# Context-Speech Bench 与全链路开源套件：总计划书

> **数据集定名：Context-Speech Bench（CSB）**——中英双语、每条对话音频均配高质量 Context 的复杂声学场景基准。
> 构成（≥10,000 条版）：**英文合成**=LibriSpeech 自混重叠+SparseLibriMix2 注噪复合流（2,985 **+扩产 2,800** train / 488 eval）；**中文合成**=AISHELL-3 自混重叠+注噪复合流（~2,000 train/400 eval，88k 源语料）+ **speech_env 单人+噪声**（1,300 train/200 eval，噪声档+候选词 Context）；**真实数据**=AMI 真实会议目标说话人（350/150，时间窗 Context）+ NoisyLibriSpeech-MUSAN（62 eval）。合计 **~9,435 train + ~1,300 eval ≈ 10,700+ 条**，真实:合成 ≈ 1:18，四种 Context 类型（三档成熟度线索 / 门控空注入 / 噪声 AGSC / 说话人时间窗），**100% 带 Context 标注**、全金标、说话人与样本双隔离（LibriSpeech train↔test-clean、AISHELL train↔test、SparseLibriMix2 段隔离）。聚焦场景：多说话人重叠+噪声（复合流核心段）、流式处理（三档成熟度线索+前缀截断）、门控（复杂区时间线真值+干净流负样本）。

> 收官目标：把"重叠→噪声→流式→RL"全流程沉淀为**开源套件 + 超大型高质量标注 Bench**，用大数据量、更长训练在三 Omni 模型上**坐实全部结论**，并整理数据/链路/模型/代码/报告，达到可发布状态。

## 一、资产盘点（决定可达规模）
| 资产 | 规模 | 用途 |
|---|---|---|
| LibriSpeech 全套（服务器本地） | **326,229 flac + 6,600 trans.txt 金标** | 干净段 + 自混重叠源（金标转写）|
| SparseLibriMix2 | 1,150 条双说话人金标重叠（仅用过 150）| 重叠核心段 |
| WHAM 噪声（parquet） | 300+ 真实噪声 | 注噪 @{0,5,10}dB |
| 现有评测集 | S2 曲线 55 / 门控流 30 / 推理 160 | 与 Context-Speech Bench 隔离保留，作回归对照 |

## 二、Context-Speech Bench 设计（高质量标注的三个升级）
1. **全金标，弃伪参考**：干净段不再截断后用 Mega-ASR 伪标注，而是**选用完整短句（≤5s）直接带 LibriSpeech 金标转写**；重叠段用 SparseLibriMix2 金标或 LibriSpeech 自混对的金标。
2. **说话人/样本双隔离**：训练流用 train/dev-clean 说话人 + SparseLibriMix2 第 150–1049 条；评测流用 **test-clean 说话人** + 第 1050–1149 条——杜绝任何源泄漏。
3. **结构化标注五元组**：每条流带 {分段金标转写（含说话人归属）、复杂区起止真值、SNR 档、三档成熟度线索（clue@2s/4s/full，SepFormer+Mega-ASR 同管线）、场景标签 mix/clean}。

**规模**：训练 **3,000 条**（1,000 SparseLibriMix2 注噪复合流 + 1,400 LibriSpeech 自混重叠注噪复合流 + 600 纯干净流）；评测 **500 条**（400 复合 + 100 干净）。约为此前 RL 数据的 **19 倍**、评测集的 16 倍。

## 三、大规模训练（chain GDPO，坐实结论）
| 模型 | episodes × epochs | 资源 | 预估 |
|---|---|---|---|
| MiniCPM-8B | 2,000 × 2 | GPU0 | ~13h |
| Qwen3-30B | 1,200 × 1 | GPU1,2（并行于 MiniCPM）| ~14h |
| Ming-104B | 600 × 1 | 4 卡（前两者完成后）| ~11h |
方法沿用全阶段 RL 报告 §2.2/2.3（Gate-then-Transcribe + 三奖励 + GDPO 解耦 + Eq.8 条件化 v2 配置——直接用张力修复后的最优配方）。总墙钟 ~28h，由 wakeup 链自动驱动，中途产出 checkpoint 与训练曲线。

## 四、评测（Context-Speech Bench-eval 500 条全 held-out）
- **M1 主表**：300 复合流 × 线索{none, t2, full} × {base, mega-RL}：GATE 准确率 / 金标 cpWER / 干净段+复杂段召回；
- **M2 回归对照**：旧 30 流 E1 + 55 条 S2 曲线集（验证与小数据结论一致且更强）；
- **M3 推理智商**：SpeakerCounting / MultiSpeakerDetection；
- 训练曲线大图 + before/after 主图。

## 五、开源套件（release_v2/omni-context-suite/）
```
omni-context-suite/
├── README.md                  # 通俗导览(中英)
├── pipeline/                  # 线索合成管线(diar/ASR/分离/VAD/SNR/门控)
├── datagen/                   # Context-Speech Bench 生成器(mega_gen.py 一键再生全部数据)
├── train/                     # SFT 训练器×3 + RL 训练器(gdpo_chain_train 等)
├── eval/                      # 全部评测器(bench/流式/门控/chain/latency)
├── rl/                        # GDPO 实现要点文档 + 奖励设计
├── data/                      # manifests + 线索 jsonl(音频由 datagen 再生)
└── checkpoints/               # 全部 LoRA(~10个)

注：报告文件不再放置在套件子目录中，统一保留在 `/cpfs_speech3/yulian.zpf/Omni-Context` 一级目录。
```

## 五·五、相关工作对照与差异化方案（IRAF, arXiv:2606.06559）

IRAF 针对全双工对话的智能体串音，在**声学嵌入层**做逐帧可靠性门控（连续标量缩放用户表征），TinyLLaMA 1.1B 栈、<200ms、MS-MARCO 合成数据。与我们的对照与应对：

**定位差异（写进各报告的"相关工作"段）**：
| | IRAF | 本工作 |
|---|---|---|
| 干扰说话人处理 | 当噪声**抑制**（只保用户）| **保留并理解**（全员 who-said-what）|
| 作用层 | 声学嵌入滤波 | 语义上下文线索 + 训练内化（可与 IRAF 级联互补）|
| 门控形态 | 逐帧连续标量（监督学习，WER 标签）| 场景级 GATE（GDPO RL，实测下游收益奖励）→ 升级软门控（见下）|
| 可证明性 | 无 | 零泄漏三重保障（静音探针/泄漏门禁/防抄契约）|
| 规模与语言 | 1.1B 专用栈、英文 | 8B–104B 三异构 Omni、中英双语 Context-Speech Bench 14,780 条 |
| 其自认局限 vs 我们 | 同声线失效 / SNR<0dB 不可靠 / 单语 | SepFormer+关键词不依赖声纹；Context-Speech Bench 含 0dB；双语 |

**借鉴转化的改进方案（体现我们优势）**：
1. **软门控（零训练成本，优先落地）**：利用 GDPO 哨兵解析策略天然输出的 π(COMPLEX) 连续概率，按置信度三档注入（≥0.8 全线索 / 0.5–0.8 仅时间窗弱线索 / <0.5 不注入），在 Context-Speech Bench-eval 上与硬门控对照——对标 IRAF 连续门控思想且方法更优雅（RL 学到的概率 vs 监督回归）。排入 P3 完成后的 P3.5 实验。
2. **分级哨兵**（文档方案）：信号级帧检测器（毫秒级粗筛）→ LLM 哨兵（0.3s 精判）→ 线索链路，写入部署蓝图。
3. **Context-Speech Bench v2 扩展项（backlog）**：duplex 串音子集（用户语音 + agent TTS 回声混合），增强对全双工场景的外部效度。

## 六、报告通俗化（与其他报告互补）
- 《GDPO全阶段强化学习_论文级报告》增写"通俗导读"框 + 术语对照 + 指向总报告/流式报告的互补引用；
- 新增《Context-Speech Bench与开源套件_总报告》：大白话讲数据怎么造、训练证明了什么、怎么用套件；
- README 重写为发布版导览。

## 七、排程
P1 数据生成（~4h，2 GPU 分片）→ P2 三模型大训练（~28h，wakeup 链）→ P3 评测（~8h）→ P4 套件打包+报告通俗化（~2h）。执行进度持续落盘本文件。
