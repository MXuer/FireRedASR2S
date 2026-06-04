# Fun-ASR-Nano-2512

References:

- https://github.com/modelscope/FunASR
- https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512

## Language Support

The pipeline catalog exposes Fun-ASR-Nano only for Chinese, English and
Japanese. Its ASR-native timestamp provider has the same language restriction.
Configure them as `zh_cn`, `en_us`, or `ja_jp`; the adapter converts them to
FunASR's native full language names.

## Environment

Use the project conda environment:

```bash
conda activate fireredasr2s
pip install tiktoken
```

`funasr==1.3.1`, `modelscope` and `huggingface_hub` are already available in
the environment. Fun-ASR-Nano downloads the model through the selected hub when
`AutoModel` is first constructed. The default adapter config uses ModelScope:

```python
FunAsrNanoConfig(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    hub="ms",
    device="cuda:0",
)
```

Run with `CUDA_VISIBLE_DEVICES=4,5,6,7`; inside the process, `cuda:0` maps to
physical GPU 4.

## Adapter

`semantic_asr.adapters.funasr_nano.FunAsrNano` normalizes FunASR output
to:

```python
{
    "uttid": "...",
    "text": "...",
    "confidence": 0,
    "timestamp": [[token, start_s, end_s], ...],
}
```

The timestamp provider is `FunAsrNanoTimestampProvider`, which validates the
ASR-native timestamps.

`FunAsrNano.transcribe` writes each VAD segment to a temporary wav path and
passes the full path list to `AutoModel.generate`, so pipeline batches are sent
as one FunASR batch instead of one generate call per segment.

Fun-ASR-Nano can emit punctuation. The adapter defaults to
`preserve_punctuation=False` for the current FireRedPunc re-punctuation
experiment. Set `preserve_punctuation=True` and use `AsrNativePunc` when keeping
FunASR native punctuation.

## Test

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python semantic_asr/examples/test_funasr_nano.py \
  --wav_path data/test/short.wav \
  --max_seconds 30 \
  --device cuda:0 \
  --hub ms
```
