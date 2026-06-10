# 最近使用 Codex 完成的复杂工作总结

## 1. 任务背景：这个任务要解决什么业务或技术问题

这轮工作围绕一个核心目标：把原始 FireRedASR2S 中“长音频中文句意识别”的经验，抽象成一个可扩展的多语种 semantic ASR pipeline。最终系统不再绑定某一套固定模型，而是支持按语言和任务组合不同的四类模块：

- VAD：检测非静音语音段；
- ASR：生成转写文本；
- Timestamp provider：提供字/词级时间戳，可以来自 ASR native timestamps，也可以来自 forced alignment；
- Punctuation：保留 ASR native punctuation，或用外部标点模型重新加标点。

业务上，这解决的是长音频、多语种、跨模型组合下的“句意级 ASR 输出”问题。输出不仅要有文本，还要能稳定生成 `json/jsonl/csv/srt/TextGrid`，并保证最终句子边界不会切到语音中间。

## 2. 原始复杂度：为什么这个任务复杂

这个任务的复杂度主要来自多个维度叠加：

- 模型组合多：FireRed VAD/Punc、Silero VAD、TEN VAD、Whisper large、Fun-ASR-Nano、Qwen3-ASR、Dolphin、Seamless M4T、Qwen3 ForcedAligner、MMS Forced Aligner、XLM-R punctuation 都要进入同一套抽象。
- 模型能力不一致：有些 ASR 自带字词时间戳和标点，例如 Fun-ASR-Nano、Dolphin；有些只给文本，需要 Qwen3 ForcedAligner 或 MMS 做强制对齐，例如 Whisper、Seamless。
- 语言输入格式不一致：用户希望统一用 `zh_cn`、`pt_br`、`ar_sa`、`ko_kr` 这类 canonical language id，但不同模型内部需要 `Chinese`、`cmn`、`pt`、`Portuguese` 等不同格式。
- 长音频切分风险高：VAD 过短会破坏句意，MMS `use_star=True` 会让 token 间时间戳不连续，`use_star=False` 又会把静音并入首尾 token，因此需要重新设计 raw VAD、ASR、MMS 和句意合并的顺序。
- 边界判断非常脆弱：仅靠标点会切到语音；仅靠 VAD silence 会把半句拆碎；仅靠 RMS valley 抗噪和跨语种稳定性不够；token timestamps 本身也可能不精确。
- 输出格式多：最终 JSON、CSV、SRT、TextGrid 都要跟 `sentences` 保持一致，不能只改 `vad_segments_ms` 而遗漏 sentence 层。
- 真实模型验证成本高：涉及 GPU、conda 环境、本地模型缓存、离线模型路径，以及多语种真实音频 smoke tests。

## 3. Codex 参与方式：Codex 具体帮我做了哪些工作

Codex 主要参与了以下几类工作。

### 代码理解和架构拆解

- 阅读并分析原始 FireRedASR2S 逻辑，将其抽象为 `semantic_asr` 项目。
- 设计并实现四阶段强制 pipeline：VAD、ASR、timestamp provider、punctuation。
- 从“一种组合一个 Python 脚本”迁移为 registry + config runner：
  - `semantic_asr/registry.py`
  - `semantic_asr/config.py`
  - `semantic_asr/run_pipeline.py`
  - `configs/*.json`
- 生成架构图：
  - `docs/architecture.drawio`

### 模型适配和能力建模

- 新增或整理多个模型 adapter、文档和测试样例：
  - `semantic_asr.adapters.ten_vad.TenVadAdapter`
  - `semantic_asr.adapters.mms_forced_aligner`
  - `semantic_asr.adapters.xlm_roberta_punctuation`
  - Qwen3-ASR、Dolphin、Seamless M4T、Fun-ASR-Nano、Whisper 等相关 adapter 和 example tests。
- 增加模型语言能力映射和查询能力：
  - `semantic_asr/language_support.py`
  - `semantic_asr/language_mapping.py`
  - `semantic_asr/query_models.py`
- 统一 canonical language id，内部自动映射到模型原生语言字段。

### 强制对齐和 timestamp 逻辑重构

- 将 MMS forced aligner 从外部 `l2s` 依赖中抽取到 `semantic_asr.mms_runtime`，避免项目外部再放一套包。
- 按用户要求取消 adapter 临时写 segment wav，改成直接用 `SpeechSegment.wav` 内存音频。
- 固定 MMS `use_star=False`，避免 `<star>` token 破坏 token 间连续时间戳。
- 将流程改为 raw VAD 后立即 ASR + MMS，之后再做句意合并，对应提交：
  - `0aaff6d align mms on raw vad segments`

### 句意边界和不切语音策略

Codex 多轮定位了 Arabic 和 Portuguese 的边界问题，并逐步演进了 `semantic_asr/sentence_boundaries.py`：

- 初版 audio-safe boundary fusion：
  - `3e52710 add audio-safe sentence boundary fusion`
- 修复 active speech boundary 被 target duration 强切：
  - `7e236ff prioritize active speech sentence safety`
- 引入 frame-level VAD probability：
  - `8fe9118 use vad probabilities for sentence boundaries`
- 将 raw VAD silence、VAD probability valley、RMS valley 从“强制切分”改为“安全可切分证据”，再叠加 semantic completeness：
  - `a006f5a Require semantic completeness for boundaries`

最近一次修复的核心函数是：

- `fuse_sentence_boundaries(...)`
- `_semantic_boundary(...)`
- `_merge_sentence_into_previous(...)`

相关调试字段也被写入 `sentence_boundary_decisions`：

- `audio_safe`
- `audio_reason`
- `semantic_complete`
- `semantic_reason`
- `speech_prob_min`
- `speech_prob_mean`
- `speech_prob_max`
- `speech_prob_boundary_ms`

### 测试、真实 smoke 和文档补齐

- 给新增模型补安装/环境文档，例如：
  - `docs/models/ten_vad.md`
  - `docs/models/xlm_roberta_punctuation.md`
  - `docs/models/mms_forced_aligner.md`
- 给新增模型补 standalone 测试样例，例如：
  - `examples/test_ten_vad.py`
  - `examples/test_xlm_roberta_punctuation.py`
  - `examples/test_mms_forced_aligner.py`
- 增加多语种测试矩阵和 smoke runner：
  - `docs/test_audio_matrix.md`
  - `examples/run_multilingual_smoke_tests.py`
- 记录真实实验结果：
  - `docs/experiments/multilingual_smoke_20260604.md`
  - `docs/sentence_boundary_fusion.md`

## 4. 我的人工决策：哪些地方是我判断、确认、取舍或最终拍板的

这轮工作里，关键方向主要由人工决策拍板：

- 确认 pipeline 强制要求四个模块：VAD、ASR、timestamp provider、punctuation。
- 指定 `semantic_asr` 后续会单独作为仓库维护，并要求把 docs/tests/examples/configs 移出 runtime package。
- 确认统一语言配置使用类似 `zh_cn` 的 canonical id，模型内部字段自动映射。
- 指定每新增一个模型都必须补安装/环境文档和单独测试样例。
- 指定使用 GPU 4、5、6、7，并确认可以联网和 GPU 命令绕开沙盒。
- 判断 Thai 不适合用 XLM-R punctuation，并要求考虑无标点语言的句意边界策略。
- 对 MMS `<star>` 的行为给出背景和取舍，并最终确认：
  - raw VAD 后立刻 ASR + MMS；
  - MMS `use_star=False` 固定；
  - 对齐后再根据标点和声学信息做句意合并。
- 指出 Portuguese raw-align 结果句意完整性变差，提供具体 JSON 片段，推动最新一轮 semantic completeness gating。
- 多次通过听感或波形检查反馈边界效果，例如 Arabic 的 `350.485s -> 350.525s`、`202.797s -> 203.037s`，以及 Portuguese 的 `47.895s-58.600s`。

## 5. 最终产出：主要代码变更、测试结果、文档、脚本或自动化成果

### 主要代码和架构成果

- 新的独立 runtime package：
  - `semantic_asr/`
- 统一 config runner：
  - `semantic_asr/run_pipeline.py`
  - `semantic_asr/config.py`
  - `semantic_asr/registry.py`
- 多语种、多模型 config：
  - `configs/ar_sa.json`
  - `configs/en_us.json`
  - `configs/hi_in.json`
  - `configs/ja_jp.json`
  - `configs/ko_kr.json`
  - `configs/pt_br.json`
  - `configs/ru_ru.json`
  - `configs/th_th.json`
  - `configs/vi_vn.json`
  - `configs/zh_cn.json`
- 核心边界融合逻辑：
  - `semantic_asr/sentence_boundaries.py`
- 输出格式支持：
  - JSON
  - JSONL
  - CSV
  - SRT
  - TextGrid

### 最近关键 commits

- `a006f5a Require semantic completeness for boundaries`
- `0aaff6d align mms on raw vad segments`
- `8fe9118 use vad probabilities for sentence boundaries`
- `6c288d9 add ten-vad`
- `dd56226 enable boundary fusion for short multilingual configs`
- `e8668f4 add multilingual five minute configs`
- `223d210 move resources outside runtime package`
- `9d348cd reorg`
- `7e236ff prioritize active speech sentence safety`
- `3e52710 add audio-safe sentence boundary fusion`

### 最近真实实验结果

Portuguese raw-align semantic-completeness historical retest：

```text
historical Portuguese semantic-completeness output JSON
```

结果：

- 37 final sentences
- 574 words
- 18,750 VAD probability frames
- sentence overlap: 0
- word overlap: 0

被修复的片段：

```text
47.895s-58.600s
O primeiro ponto que gostaria de destacaréque, ao alinhar estratégia de tecnologia e negócios, as empresas podem criar uma conexão harmoniosa entre diferentes setores.
```

### 验证命令

最近一次完整验证包括：

```bash
conda run -n fireredasr2s python -m unittest discover -s tests -p 'test_*.py'
conda run -n fireredasr2s python -m compileall -q semantic_asr tests examples
conda run -n fireredasr2s python -c "from pathlib import Path; from semantic_asr.config import load_pipeline_profile; paths=sorted(Path('configs').glob('*.json')); [load_pipeline_profile(str(path)) for path in paths]; print('parsed', len(paths), 'configs')"
git diff --check
```

验证结果：

- 49 tests passed
- `semantic_asr/tests/examples` compile passed
- 18 configs parsed
- `git diff --check` passed

## 6. 效率提升：不用 Codex 可能需要多久，用 Codex 实际用了多久

如果不用 Codex，按这个任务的实际范围估算，人工完成可能需要 2 到 4 周：

- 1 到 2 天理解原始 FireRedASR2S 长音频逻辑；
- 2 到 4 天设计 pipeline 抽象、registry、config runner；
- 3 到 6 天适配多模型、多语种和语言映射；
- 3 到 5 天处理 MMS forced alignment、`use_star`、raw VAD、输出格式一致性；
- 3 到 5 天定位边界问题、跑真实音频、对比 JSON/TextGrid/SRT；
- 2 到 4 天补测试、文档、实验报告和环境记录。

使用 Codex 后，这些工作被压缩成连续多轮协作完成。效率提升主要体现在：

- Codex 能快速跨文件理解已有设计，把变更点定位到具体模块，例如 `semantic_asr/core.py`、`semantic_asr/config.py`、`semantic_asr/sentence_boundaries.py`。
- Codex 能在每次修改后立即补测试、跑验证、更新文档，减少“代码写完但外围没跟上”的人工上下文切换。
- 对真实模型实验，Codex 能自动组织命令、检查输出 JSON、统计 overlaps、抽取指定时间段附近的 `sentence_boundary_decisions`。
- 对边界 bug，Codex 能从用户给出的片段反查 raw VAD、token gap、VAD probability、RMS valley 和 semantic punctuation，一次性给出可验证修复。

实际体感上，Codex 把大量“搜索文件、串联上下文、补样板代码、跑验证、写文档”的时间从人工身上移走了。人工主要保留在听感判断、策略拍板和模型选择上。

## 7. 质量提升：Codex 如何帮助减少遗漏、增强测试、发现潜在问题或改善可维护性

质量提升主要体现在四方面。

### 减少遗漏

- 每次新增模型都补文档和 standalone test，避免“能跑但没人知道怎么装、怎么验证”。
- 每次改 pipeline 输出，都同步检查 JSON、CSV、SRT、TextGrid 是否一致。
- 修复过一次只改 `vad_segments_ms` 但没有同步 `sentences` 的问题，后续把 sentence 层作为最终输出一致性的核心检查对象。

### 增强测试

- 增加了 fake registry/config smoke tests，降低真实模型不可用时的开发阻塞。
- 为 sentence boundary 增加单测覆盖：
  - active speech 内短边界合并；
  - VAD silence 支持时 snap boundary；
  - frame probability valley snap；
  - high speech probability 超过 `max_sentence_s` 仍不强切；
  - Portuguese reported fragment regression。
- MMS token preparation 覆盖 `zh_cn`、`ko_kr`、`ja_jp`，保证 CJK no-space scripts 的 forced alignment 输入更稳定。

### 发现潜在问题

- 发现 `PipelineConfig.from_file` 并不是实际 API，最终用 `load_pipeline_profile` 做配置解析验证。
- 发现 Thai 不在 XLM-R punctuation 支持范围内，避免错误使用 punctuation 模型。
- 发现 Fun-ASR-Nano 语言支持应限制为中文、英文、日文。
- 发现 Qwen3-ASR 语言输入需要模型自己的全拼语言名，而项目配置应保持 canonical id。
- 发现 MMS `use_star=True` 会让 token 时间戳不连续，而 `use_star=False` 又要求后续句意合并更聪明。

### 改善可维护性

- 从组合脚本迁移到 registry/config runner 后，新增组合不需要复制 Python pipeline 脚本。
- 语言映射集中在 `semantic_asr.language_mapping` 和 `semantic_asr.language_support`，避免 adapter 内到处写特殊判断。
- `sentence_boundary_decisions` 提供可解释调试字段，之后每个边界为什么 keep/merge 都能追踪。
- `docs/sentence_boundary_fusion.md` 固化了策略：不切语音第一，句意第二，长度第三。

## 8. 可复用经验：以后类似任务可以如何继续使用 Codex

后续类似任务可以继续沿用这套协作方式：

- 先让 Codex 读当前 adapter、config、test、docs，再在 `tasks/todo.md` 写明确 checklist。
- 新增模型时，让 Codex 同时完成四件事：
  - adapter；
  - language support metadata；
  - install/download/environment docs；
  - standalone output-shape test。
- 新增语言或组合时，不写新组合脚本，优先新增 `configs/*.json`，用 `semantic_asr/run_pipeline.py` 跑。
- 真实音频调试时，让 Codex 自动抽取：
  - `raw_vad_segments_ms`
  - `asr_vad_segments_ms`
  - `timestamp_segments`
  - `semantic_sentences`
  - `sentences`
  - `sentence_boundary_decisions`
  - overlap count
- 对边界问题，继续使用“用户听感/波形判断 + Codex 数据反查”的模式。人工指出可疑时间段，Codex 负责关联 VAD、token timestamp、probability、RMS 和 text punctuation。
- 每次重要策略变化都要求 Codex 更新：
  - `PROGRESS.md`
  - `docs/*.md`
  - 对应实验文档
  - 单元测试和真实 smoke 记录。

这套模式特别适合多模型、多语言、长音频这类工程：人工负责策略和质量判断，Codex 负责持续把策略落进代码、测试、文档和可复现实验里。
