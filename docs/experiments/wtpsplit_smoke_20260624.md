# wtpsplit / SaT multilingual smoke test

Date: 2026-06-24

## Goal

Evaluate whether `segment-any-text/wtpsplit` can be used as a text-level semantic sentence boundary candidate generator, especially for languages where punctuation is weak or unavailable, such as Thai.

Repository:

- https://github.com/segment-any-text/wtpsplit

Model tested:

- `segment-any-text/sat-12l-sm`

## Environment

`wtpsplit` was installed successfully in `fireredasr2s`, but importing it failed because the current `torch` package is incompatible with `transformers==4.57.6`:

```text
AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'
```

The smoke test was therefore run in the existing `qwen3-asr` environment:

```text
torch 2.6.0+cu124
transformers 4.57.6
wtpsplit 2.2.1
```

The downloaded `sat-12l-sm` snapshot did not include the tokenizer files. `wtpsplit` defaults to `facebookAI/xlm-roberta-base`, so that tokenizer was downloaded separately.

## Install / run notes

Installation command used:

```bash
no-proxy conda run -n qwen3-asr python -m pip install wtpsplit
```

Minimal usage:

```python
from wtpsplit import SaT

sat = SaT("sat-12l-sm", from_pretrained_kwargs={"local_files_only": True})
segments = sat.split(text)
```

Length-constrained usage:

```python
segments = sat.split(text, max_length=35)
```

Note: in `wtpsplit==2.2.1`, `SaT.split()` does not accept `lang_code`. The language-specific LoRA path requires both `language` and `style_or_domain`, and was not evaluated in this smoke test.

## Official language support

The official README states that SaT supports 85 languages. Thai is explicitly listed as supported:

```text
th | Thai
```

## Smoke inputs and outputs

### Chinese

Input:

```text
今天我们主要讨论项目的进展然后看一下下周的计划如果大家没有问题我们就开始第一个议题
```

Output:

```text
1. 今天我们主要讨论项目的进展然后看一下下周的计划如果大家没有问题我们就开始第一个议题
```

Observation: default segmentation did not split unpunctuated Chinese.

With `max_length=35`:

```text
1. 我们今天先看销售数据然后讨论客户反馈最后决定下周的发布计划
2. 如果没有其他问题就从第一个部分开始
```

Observation: length constraint can create acceptable boundaries, but this is more length-driven than clearly semantic.

### English

```text
1. Today we are going to review the project progress then we will discuss next weeks plan
2. if there are no questions lets start with the first topic
```

Observation: good conditional boundary.

### German

```text
1. Heute besprechen wir den Projektfortschritt
2. danach schauen wir uns den Plan fuer naechste Woche an
3. wenn es keine Fragen gibt beginnen wir mit dem ersten Thema
```

Observation: good clause/topic boundaries.

### Thai

Input:

```text
วันนี้เราจะพูดถึงความคืบหน้าของโครงการจากนั้นจะดูแผนงานของสัปดาห์หน้าถ้าไม่มีคำถามเราจะเริ่มหัวข้อแรก
```

Output:

```text
1. วันนี้เราจะพูดถึงความคืบหน้าของโครงการจากนั้นจะดูแผนงานของสัปดาห์หน้า
2. ถ้าไม่มีคำถามเราจะเริ่มหัวข้อแรก
```

Observation: Thai is usable. The split before the conditional phrase is natural.

Longer Thai with `max_length=35`:

```text
1. วันนี้เราจะพูดถึงยอดขายของเดือนนี้
2. จากนั้นจะดูความคิดเห็นจากลูกค้า
3. และสุดท้ายจะตัดสินใจแผนการเปิดตัว
4. ในสัปดาห์หน้าถ้าไม่มีคำถามเพิ่มเติม
5. เราจะเริ่มจากส่วนแรก
```

Observation: mostly useful, but one boundary is imperfect: `ในสัปดาห์หน้าถ้าไม่มีคำถามเพิ่มเติม` keeps a time phrase and conditional phrase together. Still better than having no text-level boundary model for Thai.

### Vietnamese

```text
1. Hôm nay chúng ta sẽ trao đổi về tiến độ dự án sau đó xem kế hoạch tuần tới
2. nếu không có câu hỏi thì bắt đầu chủ đề đầu tiên
```

Observation: good conditional boundary.

### Korean

```text
1. 오늘은 프로젝트 진행 상황을 이야기하고 다음 주 계획을 확인하겠습니다
2. 질문이 없으면 첫 번째 안건부터 시작하겠습니다
```

Observation: good sentence-like boundary.

### Japanese

```text
1. 今日はプロジェクトの進捗について話し合いそのあと来週の計画を確認します
2. 質問がなければ最初の議題から始めます
```

Observation: good sentence-like boundary.

### Arabic

```text
1. اليوم سنتحدث عن تقدم المشروع ثم نراجع خطة الأسبوع القادم
2. إذا لم تكن هناك أسئلة فسنبدأ بالموضوع الأول
```

Observation: good conditional boundary.

### Russian

```text
1. Сегодня мы обсудим ход проекта затем посмотрим план на следующую неделю
2. если вопросов нет начнем с первой темы
```

Observation: good conditional boundary.

## Assessment

`wtpsplit` is worth testing as a text-level boundary candidate generator.

Recommended use:

1. Run ASR and timestamp alignment as usual.
2. Run `wtpsplit` on recognized text to produce candidate semantic boundaries.
3. Map text candidates back to timestamp tokens.
4. Let the existing audio-safe boundary logic decide final cut points using VAD/speech probability and token timestamps.

Do not let `wtpsplit` directly determine audio cut points. It has no acoustic evidence and can still produce boundaries inside active speech.

## Language-specific notes

- Thai: promising. This is the most useful immediate target because Thai punctuation is weak and common punctuation restoration models often do not support Thai well.
- Chinese: default unpunctuated segmentation is weak. Use with caution, or only with length constraints as a secondary signal.
- Korean/Japanese/Vietnamese/Arabic/Russian/German/English: smoke results are reasonable.

## Open questions

- Evaluate language-specific LoRA mode with `language` and `style_or_domain`.
- Benchmark speed on long ASR transcripts.
- Test reconstruction guarantees with whitespace-sensitive languages.
- Test on real ASR outputs, not only hand-written clean samples.
- Decide whether this should live in the main environment or a sidecar environment, because `fireredasr2s` currently has a torch/transformers incompatibility for this package.

