# Fun-ASR-Nano-2512

References:

- https://github.com/modelscope/FunASR
- https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512

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

`sentence_asr_pipeline.adapters.funasr_nano.FunAsrNano` normalizes FunASR output
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

## Test

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python sentence_asr_pipeline/examples/test_funasr_nano.py \
  --wav_path data/test/conf_0002_0002_001003.wav
```
