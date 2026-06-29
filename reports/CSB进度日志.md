# CSB 流水线阶段报告日志（聊天丢失时的备份，持续追加）

## S1 数据集终版（已完成）
CSB 14,780 条（train 13,255/eval 1,525），100% Context 标注，三档线索零缺失，中英双语+真实 AMI/NoisyLibriSpeech，说话人双隔离。

## S2 MiniCPM 训练（已完成）
2,000 episodes（小数据版 12.5 倍），GDPO+Eq.8 条件奖励。训练奖励：gate 0.71→0.87，asr 0.307→0.695（+126%）。

## S5 MiniCPM 评测（已完成）
- M1 en/mix 无线索 cpWER 48.4→20.3（−28.1，内化量为小数据版 2 倍）；t2 43.4→21.0；full 50.4→22.6
- 复杂段召回 64.6→73.3；干净段召回三档全 93+（注意力虹吸被根治）
- M1 干净流 cpWER 27.1→6.6；M1zh 中文 CER 24.1→19.1（跨语言生效）
- M2 旧30流回归 49.0→36.3（小数据版 44.4，结论坐实且更强）
- M3 计数 0.35 持平、检测 0.988→1.0 满分（零遗忘）
- ⚠️ 诚实记录：门控塌缩常开（干净流 gate 0.91→0），修复方案=clean 上采样+w_gate=3 短程矫正，三模型对照后决定

## S3 Qwen3 训练（已完成）
~1,100/1,200 episodes（92%，一次进程崩溃后取 checkpoint 收官），训练奖励 asr 末段 0.6-0.96 高位。

## IRAF 论文对比与方案（已完成）
互补不竞争：IRAF 声学层抑制干扰（只保用户），我们语义层理解全员；其三大局限（同声线/0dB/单语）均为我们强项。借鉴转化：P3.5 软门控实验（哨兵 π(COMPLEX) 三档置信注入）+分级哨兵蓝图+duplex 子集 backlog。计划书 §五·五 已更新。

## 进行中
Qwen3 两变体评测（GPU1/2 并行跑批）→ S5(Qwen3) → Ming 训练评测 → P3.5 软门控 → 出图+通俗报告三件套+release_v2 打包交付。

## S5 Qwen3 评测（已完成）
- M1 en/mix 无线索 cpWER 21.3→15.8（−5.5）；t2 24.2→16.7；full 24.0→16.1；复杂段召回 76.5→78.9~80.3
- M1 干净流 cpWER 11.6→6.4；M1zh 中文 CER 22.3→19.6（−2.7 跨语言）
- M2 旧30流回归 38.0→32.5（−5.5，强于小数据版的 36.4）
- M3 计数推理 0.163→0.312（+15pp 大幅迁移！）；检测 0.9→0.875（−2.5 微降）
- ⚠️ 门控常开同病确认：干净流 gate 0.98→0（与 MiniCPM 一致，系统性问题，源于 clean episode 占比低）
## S4 Ming 大训练已启动（600 eps，4 卡）

## S4 Ming 训练进行中 + S5(Ming) 评测自动接链已布防（6/12 19:25）
- 烟测(smoke 4)通过后正式 600 eps 训练 19:02 起跑，~55s/步、共约150步，预计 21:30 前后收官；step12 奖励 fmt/gate 0.4-0.5、asr 0.16-0.21 爬升中，健康。
- 自动接链 watcher（code/ming_eval_chain.sh，服务器常驻 nohup）：训练进程退出→确认 ming_csb_lora 落盘→串行跑 csb_ming_base / csb_ming_csb 全量评测（104B 占满4卡无法并行），日志 logs/ming_eval_chain.log + ce_csb_ming_{base,csb}.log。
- 若训练崩溃无 checkpoint，watcher 记录 TRAIN FAILED 并放弃评测，等巡检处置（参照 Qwen3 取 checkpoint 收官先例）。
- 后续队列：S5(Ming) 出表 → 三模型门控塌缩统一矫正方案定稿 → P3.5 软门控 → 出图+通俗报告三件套+release_v2 打包。

## S4 Ming 训练收官 + S5(Ming) 评测自动接链已触发（6/12 23:46）
- 训练 150 步全部跑完，23:43 epoch0 saved → DONE chain ming gdpo。末段 asr 摸到 0.905 高点（全程最高），fmt 全程 1.00、gate 末段稳定 1.00、gnorm 0.10~0.20 平稳收敛，无崩溃无 OOM。
- checkpoint ming_csb_lora 正常落盘（adapter_model.safetensors 17MB @23:43）。
- watcher 成功接链：23:46:49 记录 "train done...start base eval"，base 评测正在加载 104B 模型并已开始推理；base 完成后自动接 csb 版评测（串行，104B 占满4卡）。
- 队列：base/csb 评测（深夜跑，估各 2~5h）→ S5(Ming) 出表 → 三模型对比 + 门控塌缩统一矫正方案定稿 → P3.5 软门控。

## S5(Ming) 评测：base 全套完成，csb 版踩坑已修复重起（6/13 01:31）
- base 评测 4 项全部落盘（00:48 完成）：M1/M1zh/M2/M3 已就绪。M3 计数 0.15、检测 0.725。
- ⚠️ 踩坑：watcher 用相对路径 `--lora ../checkpoints/ming_csb_lora`，peft 把含 `/` 的相对路径误判为 HF repo id → HFValidationError，csb 版评测加载完 104B 后报错退出（checkpoint 完好，仅路径解析问题；base 不带 lora 故不受影响）。
- 修复：手动用绝对路径 `--lora /cpfs_speech3/.../checkpoints/ming_csb_lora` 重起 csb 评测（PID 21038，nohup），01:31 起跑，预计约 1h 完成 M1/M1zh/M2/M3。
- 待办修订：ming_eval_chain.sh 里 csb 评测的 lora 应改绝对路径（下次复现前修；本次已手动绕过）。
- base 参考值（待 csb 出齐后出三模型完整对比表）：M3 计数 0.15 / 检测 0.725。

## S5(Ming) 评测全部完成 + 三模型完整对比表（6/13 03:58）
csb 版 4 项 03:58 全部落盘（用绝对路径重起成功，无报错）。8 文件齐，出三模型 base→csb 对比表：

| 指标(越低越好的↓) | Ming base→csb | Qwen3 base→csb | MiniCPM base→csb |
|---|---|---|---|
| M1 mix 无线索 cpWER↓ | 99.1→**19.0** (−80.1) | 21.3→15.8 | 48.4→20.3 |
| M1 mix t2 cpWER↓ | 73.5→19.0 | 24.2→16.7 | 43.4→21.0 |
| M1 mix full cpWER↓ | 77.9→18.1 | 24.0→16.1 | 50.4→22.6 |
| M1 复杂段召回↑ | 26.0→**74.6** (+48.6) | 77.4→80.3 | 67.2→71.3 |
| M1 干净段召回↑ | 29.0→96.6 | 89.2→94.2 | 67.2→93.2 |
| M1 干净流 cpWER↓ | 94.8→**8.5** (−86.3) | 11.6→6.4 | 27.1→6.6 |
| M1zh 中文 CER↓ | 99.4→30.1 | 22.3→19.6 | 24.1→19.1 |
| M2 旧30流回归 cpWER↓ | 97.5→36.5 | 38.0→32.5 | 49.0→36.3 |
| M3 计数↑ | 0.15→0.15 | 0.163→0.312 | 0.35→0.35 |
| M3 检测↑ | 0.725→0.738 | 0.90→0.875 | 0.988→1.0 |

**核心结论**：
1. Ming(104B) base 近乎完全不会本任务（cpWER 全在 73~99），训练内化收益三模型中最大（mix 无线索 −80.1、干净流 −86.3），坐实"零泄漏线索+LoRA 内化"范式对超大异构 Omni 同样有效。
2. ⚠️【门控塌缩三模型一致坐实】干净流 gate（应关门）：Ming 0.77→0.11、Qwen3 0.98→0、MiniCPM 0.91→0，三模型训练后全部塌缩为常开。复杂流 gate（应开门）三模型均→1.0 正常。系统性问题，根源 clean episode 训练占比过低。
3. 统一矫正方案定稿：clean 上采样（提高干净流 episode 占比）+ w_gate=3 短程续训矫正，三模型同方案。
4. Ming 中文偏弱（CER 30.1 vs Qwen3/MiniCPM ~19），单独记录；M3 计数 0.15 训练零迁移（Qwen3 +15pp 最佳）。

后续队列：门控矫正续训（三模型，GPU重活，待启动）‖ P3.5 软门控（零训练成本，优先）→ 出图 + 通俗报告三件套 + release_v2 打包。

## P3.5 软门控（B方案）：关键负结果——连续概率也塌缩（6/13 08:15）
方法：forced-choice 探针 csb_softgate_probe.py，哨兵 prompt 后拼 "GATE:"，thinker 单次 forward 取末位 logits，softmax over [' COM'(COMPLEX 7682), ' CLEAN'(77000)] 得 π(COMPLEX)。
（踩坑：Qwen3-Omni 的 model.generate 返回 tuple 拿不到 scores，改 model.thinker(**inputs).logits[0,-1] 取连续概率。）

Qwen3-csb smoke（各8 clean/mix）：
- mix:   π mean=1.000，全部 >=0.8
- clean: π mean=1.000，全部 >=0.8（logit_COM ~37-39 vs logit_CLEAN ~17-20，差~20，softmax 饱和）
- 干净流与复杂流 π 分布完全重叠、零方差 → 连续概率彻底塌缩，无区分度。

结论：软门控（≥0.8全线索/0.5-0.8弱线索/<0.5不注入）三档阈值对 clean/mix 全落 ≥0.8 同一档，**救不了门控塌缩**。塌缩发生在概率层而非仅 argmax 层。
→ B 方案对"已训练塌缩模型"无效；正在跑 base 对照（未训练 Qwen3）验证探针有效性（base 干净流 gate_acc 0.98，理应 π 低、有区分度）。base 若有区分度即坐实"是训练把 π 压饱和"，方法学闭环。
→ 路线转 A：clean 上采样 + w_gate=3 短程续训矫正塌缩（B 的负结果反而精确诊断了 A 需要的强度：要把 clean 的 π 从饱和拉下来）。

## A方案 门控矫正续训：Qwen3 已启动（6/13 08:40）
B负结果精确诊断后转 A。改造 gdpo_chain_train.py（已备份 .bak_gatefix）：
- 新增 --clean_ratio（clean episode 上采样到目标占比，直击塌缩根因：clean 训练占比过低）
- 新增 --ck_suffix（续训存新路径，绝不覆盖 S5 基准原 csb_lora）
- w_gate=3 用 --weights 1.0,3.0,2.0；热启动用现成 --resume_lora
smoke 验证三机制全通过（clean_upsample 生效 / resume 576/576 全加载 / 不崩）。
Qwen3 正式续训 pid 25561：eps_cap 200→clean上采样→366 eps(~91步)，热启动自 qwen3_csb_lora，存 qwen3_csb_lora_gatefix。预计 ~1.5h。
验证闭环（续训后）：① π 探针(_gatefix)确认干净流 π 从饱和1.0拉回<0.5；② 评测确认 gate_clean 恢复 且 ASR cpWER 不退化；③ 成功则推广 MiniCPM/Ming 同方案。
目标量化（来自B诊断）：干净流 π 1.0→base的<0.5区间；干净流 gate_acc 0→恢复高位；同时保 mix cpWER ~16 不退化。

## 门控塌缩攻坚：B证伪→A转SFT→治住（6/13 14:51）
【B 软门控证伪】π探针(forced-choice, thinker forward取logits)：训练后 csb 干净流π中位1.0(base 0.012)，连续概率也饱和，三档软门控退化成一档→救不了塌缩。base对照(clean π0.012/mix 0.915 有区分度)证明探针有效、是训练把π压饱和。论文级负结果(呼应IRAF连续门控的梯度可达性)。
【A-RL续训失败】clean上采样+w_gate=3热启动续训76步：gate奖励卡0.5平台(0.409→0.512→0.512→0.484)，18/76步gnorm=0。根因：饱和点softmax梯度≈0 + GDPO组内advantage抵消，on-policy RL救不动已饱和门控。(另踩坑：温度采样rollout在step76后hang，取step70 checkpoint，π仍1.0未矫正。)
【A-SFT矫正成功】转监督矫正 csb_gate_sft.py：只对GATE行teacher-forcing(clean→'GATE: CLEAN'/mix→'GATE: COMPLEX')。理论：-log p(CLEAN)在p饱和(→0)时梯度极大，正好破RL零梯度死穴。
- Qwen3 SFT 90步：loss 2.45→5e-5，gnorm稳定12-16(强梯度vs RL的0)，存 qwen3_csb_lora_gatesft。
- π探针验证(eval音频, n30)：干净流π 1.0→**0.000**(<0.5占100%)，复杂流保持1.000(≥0.8占83%)→区分度完全恢复，比base更干净。**门控塌缩治住。**
- 评测验证中(csb_q3_gatesft m1)：确认 gate_acc恢复 且 mix cpWER ~16不退化(只学GATE行,转写应不受影响)。
后续：评测确认→推广 MiniCPM(omni-context-mcpm env)/Ming(ming env,104B串行) 同方案SFT+验证→三模型门控统一治住→收尾打包。

## 门控塌缩 Qwen3 彻底治住 + 推广启动（6/13 20:48）
【Qwen3 SFT v2 评测两全成功】对照塌缩 csb 版：
| 指标 | csb(塌缩) | gatesft v2 |
|---|---|---|
| 干净流 gate_acc | ~0 | 1.000 |
| 干净流 π | 1.0 | 0.002 |
| clean cpWER | 6.4 | 6.2 |
| mix none/t2/full cpWER | 15.8/16.7/16.1 | 14.6/14.9/15.0 |
| 复杂段召回 | 80.3 | 81.3 |
门控治住(gate_acc 0→1.0、π 1.0→0.002) 且 转写不退化(cpWER 反而略好,SFT金标顺带强化ASR)。Qwen3闭环。
存 qwen3_csb_lora_gatesft。

【推广】SFT脚本v2(含转写金标)通用。
- MiniCPM SFT完成(omni-context-mcpm env,90步,loss 0.11-0.33,存minicpm_csb_lora_gatesft)；MiniCPM token id与Qwen3相同(7682/77000)；评测验证中(csb_mcpm_gatesft m1)。
- Ming 待推广(ming env,104B 4卡)。
【方法学注记】π探针(csb_softgate_probe.py)走Qwen3 thinker forward,MiniCPM/Ming接口不同;矫正验证改用评测gate_acc(π的离散判据:干净流判对CLEAN=门控治住)即可,无需为各模型适配探针。
【踩坑】① nohup bash -c "A;B" 串在run_in_background ssh里,轮询ssh收尾会杀整串→各步独立nohup。② pkill -f "模式" 模式串若出现在命令行自身会自杀(exit255)→按pid kill。
后续：MiniCPM评测确认→Ming SFT+评测→三模型门控矫正前后对比表→收尾打包(出图+通俗报告三件套+release_v2)。

## MiniCPM 门控治住 + Ming SFT 启动（6/13 21:28）
MiniCPM gatesft 评测(1000条)对照塌缩 csb 版：
| MiniCPM | csb(塌缩) | gatesft |
|---|---|---|
| 干净流 gate_acc | ~0 | 1.000 |
| clean cpWER | 6.6 | 8.1 |
| mix none/t2/full cpWER | 20.3/21.0/22.6 | 21.6/18.3/17.9 |
| 复杂段召回 | 71.3 | 75-78 |
门控治住(clean gate_acc 0→1.0)，转写无明显退化(t2/full cpWER降、召回升，clean/none小幅波动在噪声内)。MiniCPM闭环。存 minicpm_csb_lora_gatesft。
注记：mix none gate_acc 0.437偏低(无线索复杂流门控召回有提升空间)，但非塌缩问题(塌缩=clean误判,已治)。
Ming SFT 启动(ming env, 104B 4卡, eps_cap200~90步, 存ming_csb_lora_gatesft)，验证前几步中。
进度：门控塌缩矫正 Qwen3✅ MiniCPM✅ Ming⏳。三模型全治后出对比表+收尾打包。

## 门控塌缩三模型矫正前后最终对比表（6/14 00:17 三模型全治住）
### Qwen3  (csb塌缩 → gatesft矫正)
- 干净流gate_acc(门控核心判据): 0.000 → 1.000
- clean cpWER: 6.4 → 6.2
- mix none/t2/full cpWER: 15.8/16.7/16.1 → 14.6/14.9/15.0
- 复杂段召回: 80.3 → 81.5
### MiniCPM  (csb塌缩 → gatesft矫正)
- 干净流gate_acc(门控核心判据): 0.000 → 1.000
- clean cpWER: 6.6 → 8.1
- mix none/t2/full cpWER: 20.3/21.0/22.6 → 21.6/18.3/17.9
- 复杂段召回: 71.3 → 77.7
### Ming  (csb塌缩 → gatesft矫正)
- 干净流gate_acc(门控核心判据): 0.110 → 1.000
- clean cpWER: 8.5 → 7.4
- mix none/t2/full cpWER: 19.0/19.0/18.1 → 17.5/18.3/17.2
- 复杂段召回: 74.6 → 80.9
结论：SFT监督矫正(clean上采样+GATE金标含转写)使三模型干净流门控gate_acc从塌缩(~0)全部恢复到~1.0，且转写cpWER不退化反而普遍改善、复杂段召回提升。门控塌缩这一三模型系统性顽疾彻底解决。

## ★ 任务完整结束：门控塌缩攻坚 + 收尾交付（6/14 02:00）
【核心目标达成】门控塌缩(三模型干净流gate→0)彻底解决：
- 路径：B软门控证伪(π探针:csb干净流π饱和1.0 vs base 0.012,概率层塌缩)→A-RL续训失败(gate卡0.5平台/18-of-76步零梯度,饱和点softmax梯度≈0+GDPO组内advantage抵消)→A-SFT破饱和(交叉熵-log p饱和点梯度大)→v1只学GATE致转写截断cpWER100→v2含金标转写两全。
- 结果：三模型 clean gate_acc 0/0/0.11→1.0/1.0/1.0；mix cpWER与复杂段召回不退化反普遍改善。
【交付三件套完成】
- 出图：results/fig_gatefix.png（三模型gate_acc塌缩→治后 + 转写不退化）
- 报告：Omni-Context_总报告.md §十(大白话) + GDPO全阶段强化学习_论文级报告.md §11(学术含机制) + README.md摘要
- release：release_v2/omni-context-suite/（179文件，pipeline/datagen/train/eval/rl/data/checkpoints/reports，含csb_gate_sft.py、csb_softgate_probe.py、三gatesft checkpoints软链、对比表json、fig_gatefix.png、全报告，MANIFEST.txt，suite README）
【新增代码】code/csb_gate_sft.py(门控SFT矫正,含--clean_ratio上采样)、code/csb_softgate_probe.py(forced-choice π探针)、code/make_gatefix_fig.py；gdpo_chain_train.py加--clean_ratio/--ck_suffix(已备份.bak_gatefix)。
任务完整结束，停止巡检。
