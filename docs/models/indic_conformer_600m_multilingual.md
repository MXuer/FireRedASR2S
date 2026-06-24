# IndicConformer 600M Multilingual

Source: https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual

Status: downloaded and smoke-tested, not yet registered as a pipeline adapter.

## Environment

Model snapshot:

```text
~/.cache/huggingface/hub/models--ai4bharat--indic-conformer-600m-multilingual/snapshots/e9b71b369c048e2c6b634d4c131061c34e441179
```

The model card recommends:

```bash
pip install transformers torchaudio "onnxruntime==1.20.1" "onnx==1.20.1" "onnxruntime-gpu==1.20.1"
```

Current `fireredasr2s` has:

```text
onnxruntime: 1.23.2
providers: AzureExecutionProvider, CPUExecutionProvider
onnx: not installed
```

So the current environment can run CPU ONNXRuntime smoke tests, but it is not
the recommended GPU setup.

## Local Source Findings

The HF `auto_map` points to `model_onnx.py`. That file supports single-item CTC
and RNNT decoding.

`model_onnx_1b_batched_rnnt.py` contains a `batched_forward(..., decoding="rnnt")`
path, but the current downloaded assets do not match that file: it expects
`assets/rnnt_decoder_embed.onnx` and `assets/rnnt_decoder_rnn.onnx`, while the
snapshot contains `assets/rnnt_decoder.onnx`.

Also, the custom `from_pretrained()` implementation treats a local snapshot path
as a Hugging Face repo id. A local adapter should instantiate the local classes
directly with:

```python
IndicASRModel(IndicASRConfig(ts_folder=local_snapshot_path))
```

## Capabilities

- Languages: `as`, `bn`, `brx`, `doi`, `gu`, `hi`, `kn`, `kok`, `ks`, `mai`,
  `ml`, `mni`, `mr`, `ne`, `or`, `pa`, `sa`, `sat`, `sd`, `ta`, `te`, `ur`
- RNNT: yes, single-item local smoke passed on 5 seconds of Hindi audio
- CTC: yes, documented by the model card
- Batch inference: not currently usable with the downloaded assets; the batch
  RNNT source exists but expects different ONNX component files
- Native timestamps: CTC path can compute timestamps; RNNT path returns text only
- Native punctuation / ITN: not documented; treat as no until domain tests prove
  otherwise

## Smoke Result

Single-item RNNT smoke passed in `fireredasr2s` with CPU ONNXRuntime:

```text
सभी को शुभ संध्या आज माइक और मैं हमारे जीवन के महत्वपूर्ण पहलू
```

Use a proper `onnxruntime-gpu==1.20.1` environment before performance testing.
