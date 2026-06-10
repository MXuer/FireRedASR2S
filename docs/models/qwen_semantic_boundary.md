# Qwen Semantic Boundary

Registry name: `qwen_semantic_boundary`

Role: `punc`

This component uses the local Qwen3.6 chat API as a semantic sentence-boundary
selector. It does not restore punctuation and does not rewrite ASR text. The
model only returns inclusive token end indexes, and the pipeline reconstructs
sentence text from the original timestamp tokens.

## API

Default endpoint:

```text
http://10.10.23.16:18000/v1/chat/completions
```

The service must be called with:

```json
{
  "response_format": {
    "type": "json_object"
  },
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

Plain prompts and `/no_think` were observed to emit visible thinking text, so
they are not safe for programmatic JSON parsing.

`response_format={"type":"json_object"}` is enabled by default as an additional
OpenAI-compatible constraint. It improves the chance of parseable JSON, but it
does not replace client-side validation. Set `response_format_json=false` only
when a compatible backend rejects the field.

## Output Contract

The prompt asks Qwen to return only:

```json
{
  "token_count": 103,
  "end_indices": [102]
}
```

Client validation requires:

- `token_count` equals the number of timestamp tokens sent to Qwen.
- `end_indices` are integers.
- indexes are strictly increasing.
- the last index is `token_count - 1`.
- all tokens are covered exactly once.

The adapter also accepts continuous `spans` as a compatibility format, but the
preferred protocol is `end_indices`.

## Retry And Fallback

If Qwen returns malformed JSON or invalid indexes, the adapter retries with the
validation error included in the prompt. If all attempts fail, it falls back to
local duration-based candidate boundaries. Final audio safety is still handled
later by sentence-boundary fusion with VAD speech probability and raw VAD
silence.

## Language Support

The index-only protocol is language-agnostic. It should work for Thai and other
languages whose text Qwen can understand, because the model is not asked to
generate final text. Quality still needs per-language and per-domain smoke
testing, especially for noisy ASR, code-switching, dialects, or scripts without
spaces.

## Example Config

See `configs/th_th_qwen_boundary.json`.
