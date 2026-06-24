# Cohere Transcribe 03 2026

Source: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026

Status: downloaded and smoke-tested, not yet registered as a pipeline adapter.

## Environment

Model snapshot:

```text
~/.cache/huggingface/hub/models--CohereLabs--cohere-transcribe-03-2026/snapshots/b1eacc2686a3d08ceaae5f24a88b1d519620bc09
```

The model card recommends `transformers>=5.4.0` and says testing used
`torch==2.10.0`. Current `fireredasr2s` is:

```text
torch: 2.1.0+cu118
transformers: 4.57.6
```

## Compatibility Notes

Lightweight config and processor loading works in `fireredasr2s` when
`HF_MODULES_CACHE` points to a writable path:

```bash
HF_MODULES_CACHE=/tmp/hf_modules python -c "..."
```

Real generation in current `fireredasr2s` needs two workarounds:

- Use `torch.bfloat16`; `float16` overflows when masking attention scores.
- Add `decoder_attention_mask = torch.ones((batch, 1), dtype=torch.long)` before
  calling `generate()`.

With those workarounds, a 2-item English batch generated two outputs on GPU4.
However, this should still be treated as a compatibility smoke only. The proper
adapter should run in a newer independent environment matching the model card.

## Capabilities

- Languages: `ar`, `de`, `el`, `en`, `es`, `fr`, `it`, `ja`, `ko`, `nl`, `pl`,
  `pt`, `vi`, `zh`
- Batch inference: yes, verified with a 2-item local batch
- Native timestamps: no
- Native punctuation: yes
- Punctuation control: model card documents `punctuation=False`, but this did
  not visibly change the tested English output in the current compatibility
  smoke; retest in the final independent environment.

## Adapter Guidance

Use a sidecar/independent conda environment first. Do not pin this model to the
main `fireredasr2s` runtime until the newer Transformers stack is validated
against the rest of the pipeline.
