# Hunyuan-MT Translation Integration Design

## Goal

Add optional translation for completed ASR jobs so users can review either the
original transcript, translated text, or both in the web demo. Translation must
not change sentence boundaries, timestamps, cut segments, CSV/SRT/TextGrid
exports, or the ASR pipeline itself.

## Model Choice

Recommended first deployment:

- `Hunyuan-MT-7B-fp8` or local FP8/INT8 quantized equivalent if GPU memory is a
  concern.
- `Hunyuan-MT-7B` BF16 if there is enough GPU memory and quality is more
  important than throughput.

Do not start with `Hunyuan-MT-Chimera-7B` for the interactive web demo. Chimera
is an ensemble/refinement model intended to combine multiple translation
outputs, so it is better suited for a later "high quality/offline" mode. The
interactive UI needs predictable latency and one translation per sentence.

Official model notes from the Hunyuan-MT repository:

- Hunyuan-MT includes `Hunyuan-MT-7B` and `Hunyuan-MT-Chimera`.
- `Hunyuan-MT-7B` is the direct translation model.
- `Hunyuan-MT-Chimera` refines multiple candidate translations into one higher
  quality output.
- The repository lists normal and FP8 model links.
- The model supports translation across the current project languages such as
  Chinese, English, Japanese, Korean, Arabic, Russian, Thai, German,
  Vietnamese, Hindi and Portuguese. It also lists Cantonese (`yue`).

Sources:

- https://github.com/Tencent-Hunyuan/Hunyuan-MT
- https://arxiv.org/abs/2509.05209

## Deployment

Run Hunyuan-MT as a separate OpenAI-compatible service, not inside the ASR
worker process.

Reasons:

- ASR workers already load VAD/ASR/MMS/punctuation models and may occupy GPU
  memory for long periods.
- Translation is optional and can be scaled independently.
- The web UI can request translation after an ASR job succeeds.
- A failed translation should not invalidate an ASR result.

Recommended v1 deployment shape:

```bash
export MODEL_PATH=/path/to/downloaded/Hunyuan-MT-7B-fp8
CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 10087 \
  --trust-remote-code \
  --model "${MODEL_PATH}" \
  --served-model-name hunyuan-mt \
  --dtype bfloat16
```

If using official quantized variants, follow the model card/repository
requirements for the corresponding quantization flags. The Hunyuan-MT
repository says vLLM `0.10.0+` is expected and documents OpenAI-compatible API
deployment. Keep this environment separate from `fireredasr2s` if dependency
versions conflict.

Recommended GPU allocation:

- Keep current ASR demo workers on GPU 7.
- Put translation on GPU 6 first.
- If translation and ASR contend for memory, keep translation as a separate
service and reduce translation concurrency.

### Current Local Smoke Setup

The downloaded model path is:

```text
/home/duhu/.cache/huggingface/hub/models--tencent--Hunyuan-MT-7B-fp8/snapshots/81e5a3f7199524570ba75e61360e990ba88665e4
```

`fireredasr2s` does not include vLLM, and its transformers/torch versions are
not compatible with the model. The local `llm` conda environment was updated to
`transformers==4.56.0` and `compressed-tensors==0.11.0`, matching the model
card requirement. This conflicts with the installed LLaMAFactory package in
that environment, so use the environment for the Hunyuan-MT service only.

Current working startup command:

```bash
export MODEL_PATH=/home/duhu/.cache/huggingface/hub/models--tencent--Hunyuan-MT-7B-fp8/snapshots/81e5a3f7199524570ba75e61360e990ba88665e4
CUDA_VISIBLE_DEVICES=6 conda run -n llm python -m semantic_asr_service.hunyuan_mt_server \
  --model-path "${MODEL_PATH}" \
  --runtime-model-path service_data/hunyuan_mt_fp8_patched \
  --served-model-name hunyuan-mt \
  --host 127.0.0.1 \
  --port 10087
```

The service listens only on `127.0.0.1:10087`; the public browser talks to the
normal Semantic ASR service on `10086`, and that server calls translation
locally.

Smoke result:

- Direct `/v1/chat/completions`: `It is on the house.` -> `这顿饭由我们公司来买单。`
- Semantic ASR translation API short job:
  - `감회가 새롭습니다.` -> `再次重温这些往事，心中依然充满了感慨。`
  - `오늘은 날씨가 좋습니다.` -> `今天天气很好。`

The first long Korean demo job was too slow with the synchronous
sentence-by-sentence translation path. The current service translates completed
ASR sentences in configurable batches and keeps per-batch fallback to
sentence-by-sentence translation when the model output is malformed.

## Prompt Policy

Use batched JSON prompting for completed ASR results. Each request contains up
to `SEMANTIC_ASR_TRANSLATION_BATCH_SIZE` sentence records:

```json
[
  {"index": 0, "text": "감회가 새롭습니다."},
  {"index": 1, "text": "오늘은 날씨가 좋습니다."}
]
```

The model must return only a JSON array with the same `index` values and one
`translation` field per item. The client validates the returned indexes before
accepting the batch. If parsing or validation fails, only that batch falls back
to one-sentence-at-a-time translation.

For Chinese as target or source, the batch prompt shape is:

```text
把下面 JSON 数组中的 text 翻译成<target_language>，不要额外解释。只返回 JSON 数组，每一项包含 index 和 translation，index 必须保持不变。

<source_json_array>
```

For non-Chinese to non-Chinese, the batch prompt shape is:

```text
Translate each text field in the following JSON array into <target_language>. Return only a JSON array. Each item must contain index and translation, and index must remain unchanged.

<source_json_array>
```

Use deterministic-ish settings for review consistency, even if official
examples use sampling:

```json
{
  "temperature": 0.2,
  "top_p": 0.6,
  "max_tokens": 512
}
```

If quality is too flat, raise `temperature` toward the official example
(`0.7`). For UI review, consistency is usually more important.

## Backend Integration

Add a translation layer beside `semantic_asr_service`, not inside
`semantic_asr.core`.

New files:

- `semantic_asr_service/translation.py`
  - OpenAI-compatible client.
  - language-name mapping from project profile ids such as `ko_kr` or `ar_sa`
    to Hunyuan prompt names such as `Korean` or `Arabic`.
  - sentence translation and cache helpers.
- `semantic_asr_service/translation_worker.py`
  - Optional async worker loop for long jobs.
- `semantic_asr_service/translation_store.py`
  - Optional SQLite table for translation job status.

Recommended API:

```http
POST /v1/jobs/{job_id}/translations
{
  "target_language": "zh_cn"
}
```

Returns:

```json
{
  "job_id": "...",
  "target_language": "zh_cn",
  "status": "queued"
}
```

Polling:

```http
GET /v1/jobs/{job_id}/translations/zh_cn
```

Returns either status or completed translations:

```json
{
  "job_id": "...",
  "source_language": "ko_kr",
  "target_language": "zh_cn",
  "status": "succeeded",
  "sentences": [
    {
      "index": 0,
      "start_ms": 23660,
      "end_ms": 24961,
      "cut_start_ms": 23660,
      "cut_end_ms": 24961,
      "text": "감회가 새롭습니다.",
      "translation": "感慨万千。"
    }
  ]
}
```

Cache output under the existing job output directory:

```text
service_data/jobs/{job_id}/outputs/translations/zh_cn.json
```

The cache should include:

- model name/version
- source language
- target language
- source ASR JSON mtime or hash
- sentence count
- sentence index, timestamps, original text and translation

If the ASR JSON changes or sentence count/hash changes, invalidate the cache.

## Web UI

Add a target-language selector in the Review panel:

- `Original`
- `Translation`
- `Bilingual`

Flow:

1. User opens a completed job with `View`.
2. UI loads original JSON as today.
3. User selects target language and clicks `Translate`.
4. UI calls `POST /v1/jobs/{job_id}/translations`.
5. UI polls `GET /v1/jobs/{job_id}/translations/{target_language}`.
6. When done, the segment list uses the selected display mode:
   - original text only;
   - translated text only;
   - original + translated.

Do not redraw or modify waveform intervals based on translation. Translation is
display text only.

## Implementation TODO

1. Add translation settings:
   - `SEMANTIC_ASR_TRANSLATION_BASE_URL`
   - `SEMANTIC_ASR_TRANSLATION_MODEL`
   - `SEMANTIC_ASR_TRANSLATION_API_KEY`
   - `SEMANTIC_ASR_TRANSLATION_MAX_CONCURRENCY`
   - `SEMANTIC_ASR_TRANSLATION_TARGETS`
2. Add language mapping from profile ids to Hunyuan language names.
3. Add translation client and cache format.
4. Add async translation job store/worker.
5. Add API routes for create/poll translation.
6. Add web Review controls for original/translation/bilingual display.
7. Add tests with a fake OpenAI-compatible translation server/client.
8. After the model is downloaded, add a standalone real-model smoke test and
   document the exact install/deployment commands.
