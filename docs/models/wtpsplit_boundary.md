# wtpsplit Boundary Sidecar

Source: https://github.com/segment-any-text/wtpsplit

Component: `wtpsplit_boundary`

## Purpose

`wtpsplit_boundary` is a semantic boundary strategy, not a punctuation
restoration model. It calls a local wtpsplit/SaT sidecar and returns sentence
span candidates without rewriting ASR text.

This is especially useful for Thai, where punctuation is often weak or absent.
The same component can be tested for other languages supported by SaT.

## Environment

The current `fireredasr2s` environment cannot import `wtpsplit` cleanly because
of the local `torch`/`transformers` compatibility issue recorded in
`docs/experiments/wtpsplit_smoke_20260624.md`.

Run the sidecar from an environment that can import wtpsplit, for example:

```bash
conda activate qwen3-asr
python -m semantic_asr.sidecars.wtpsplit_server --host 127.0.0.1 --port 11001
```

The model is lazy-loaded on the first `/v1/boundaries` request.

## GPU Use

GPU is not required for the current pipeline use. wtpsplit/SaT is a text
boundary model, and the sidecar only receives ASR text/tokens, not audio.

Recommended default:

- run wtpsplit on CPU or a low-priority shared device;
- do not reserve ASR GPUs only for wtpsplit;
- consider GPU only if long transcripts or many concurrent requests make the
  sidecar a measured bottleneck.

## API

```text
GET /health
POST /v1/boundaries
```

Request:

```json
{
  "language": "th_th",
  "tokens": ["วันนี้", "ฝนตก", "เจ้าหน้าที่", "เตือนภัย"],
  "text": "วันนี้ฝนตกเจ้าหน้าที่เตือนภัย",
  "max_length": 35
}
```

Response:

```json
{
  "language": "th_th",
  "token_count": 4,
  "end_indices": [1, 3],
  "segments": ["วันนี้ฝนตก", "เจ้าหน้าที่เตือนภัย"]
}
```

## Pipeline Behavior

The adapter converts `end_indices` into `punc_sentences` with:

```json
{
  "semantic_boundary": true,
  "boundary_source": "wtpsplit"
}
```

The text is not modified and no fake punctuation is added. Sentence-boundary
fusion treats the tag as a semantic-complete candidate, but final cutting still
depends on VAD/speech-probability safety and duration rules.

## Thai Profile

`configs/th_th_whisper_th.json` uses:

```json
{
  "punc": {
    "name": "wtpsplit_boundary",
    "params": {
      "base_url": "http://127.0.0.1:11001",
      "max_length": 35,
      "min_span_s": 2.0
    }
  }
}
```
