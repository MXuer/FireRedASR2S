# TEN VAD

Reference: https://github.com/TEN-framework/ten-vad

## Environment

Use the project conda environment:

```bash
conda activate fireredasr2s
git clone git@github.com:TEN-framework/ten-vad.git
cd ten-vad
python setup.py install
```

The adapter expects 16 kHz wav input. The current project pipeline test clips
already use 16 kHz audio.

## Adapter

`semantic_asr.adapters.ten_vad.TenVadAdapter` normalizes TEN VAD output to:

```python
{"timestamps": [(start_s, end_s), ...]}
```

The adapter follows the post-processing logic in `ten_vad_utils.py`: TEN VAD
frame probabilities are converted into speech intervals with a threshold,
hysteresis, minimum speech duration and minimum silence duration.

## Test

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python examples/test_ten_vad.py \
  --wav_path data/test/short/pt_br-short.wav \
  --max_seconds 30
```

