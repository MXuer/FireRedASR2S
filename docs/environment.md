# Runtime Environment

This document records the current machine baseline and model-environment notes
for future migration.

## Machine Baseline

Collected on 2026-06-24.

```text
host: ac-wlcb-gpu-8
os: Ubuntu 20.04.6 LTS
kernel: 5.4.0-216-generic
gcc: 9.4.0
conda: 23.7.4
```

## GPU / CUDA

```text
GPU: 8 x NVIDIA A40
VRAM per GPU: 46068 MiB
NVIDIA driver: 555.42.02
nvidia-smi CUDA: 12.5
nvcc: 12.5.40
```

Typical local policy:

- ASR workers usually run on GPUs `6,7`.
- Translation usually runs on GPU `5`.
- Test jobs often use GPU `4`.
- Always set `CUDA_VISIBLE_DEVICES` explicitly for smoke tests.

## Main Environment: `fireredasr2s`

```text
python: 3.10
torch: 2.1.0+cu118
torchaudio: 2.1.0+cu118
transformers: 4.57.6
onnxruntime: 1.23.2
onnxruntime providers: AzureExecutionProvider, CPUExecutionProvider
```

Important compatibility notes:

- Import `semantic_asr` before direct Transformers imports when using this
  environment; it patches the old torch pytree API expected by
  `transformers==4.57.6`.
- `transformers==4.57.6` refuses to load PyTorch `.bin` checkpoints with
  `torch<2.6` because of the `torch.load` CVE check. Safetensors checkpoints
  are not affected.
- `onnxruntime-gpu` is not available in this environment.

Installed during the 2026-06-24 model smoke:

```text
cadence-punctuation==1.1.0
```

## Auxiliary Environment: `qwen3-asr`

Candidate newer main runtime used to verify PyTorch `.bin` checkpoints and
newer ASR models requiring torch>=2.6.

```text
python: 3.12.13
torch: 2.6.0+cu124
torchaudio: 2.6.0+cu124
torchvision: 0.21.0+cu124
transformers: 4.57.6
onnxruntime-gpu: 1.22.0
onnxruntime providers: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
```

Installed / verified model packages:

```text
cadence-punctuation: 1.1.0
dataoceanai-dolphin: 20260513
funasr: 1.3.1
gigaam: 0.1.0
kaldi-native-fbank: 1.22.3
modelscope: 1.35.4
openai-whisper: 20250625
punctuators: 0.0.7
qwen-asr: 0.0.6
qwen-omni-utils: 0.0.9
tiktoken: 0.13.0
triton: 3.2.0
```

Compatibility notes:

- FireRed VAD, FireRedASR2-AED and FireRedPunc import and CPU model-load in
  this environment after installing `kaldi-native-fbank`.
- `onnxruntime-gpu==1.27.0` looked for `libcudart.so.13` and did not match the
  local CUDA 12.4 PyTorch stack. Use `onnxruntime-gpu==1.22.0`.
- NeMo is installed in `qwen3-asr`, and `nvidia_ar_fastconformer` was
  runtime-verified there with RNNT decoding and `timestamps=True`.
- NeMo is not installed in `fireredasr2s`; install it there or run this adapter
  through a separate environment/service before using the profile from the main
  runtime.
- Lightweight validation passed on 2026-06-24:
  `tests.test_adapter_inputs`, `tests.test_mms_forced_aligner`,
  `tests.test_punctuation_strategies`, `tests.test_config_runner`,
  `tests.test_language_support`, and `tests.test_outputs` (110 tests).

## New Environment: `semantic-asr`

Created for omniASR exploration:

```bash
no-proxy conda create -y -n semantic-asr python=3.10
```

The `omnilingual-asr` installation was intentionally stopped because the user
will finish installing it manually. During the partial install, pip selected
large CUDA 12 wheels such as `torch`, `nvidia-cudnn-cu12`, and
`nvidia-cublas-cu12`; reserve enough download time and disk space when
recreating this environment.

## Download Rule

Do not run large downloads through the proxy. Use:

```bash
no-proxy <command>
```

or clear proxy variables manually if `no-proxy` is unavailable.
