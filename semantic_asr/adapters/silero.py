from dataclasses import dataclass


@dataclass
class SileroVadConfig:
    sampling_rate: int = 16000
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30


class SileroVad:
    def __init__(self, config: SileroVadConfig | None = None):
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
        import torch

        self.config = config or SileroVadConfig()
        self.model = load_silero_vad()
        self.read_audio = read_audio
        self.get_speech_timestamps = get_speech_timestamps
        self.torch = torch

    def detect(self, wav_path: str):
        wav = self.read_audio(wav_path, sampling_rate=self.config.sampling_rate)
        probabilities = self._frame_probabilities(wav)
        timestamps = self.get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=self.config.sampling_rate,
            threshold=self.config.threshold,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
            return_seconds=True,
        )
        segments = [(float(item["start"]), float(item["end"])) for item in timestamps]
        return {
            "timestamps": segments,
            "frame_speech_probs": {
                "frame_shift_ms": 32,
                "frame_length_ms": 32,
                "probs": probabilities,
            },
        }

    def _frame_probabilities(self, wav) -> list[float]:
        window_size_samples = 512 if self.config.sampling_rate == 16000 else 256
        self.model.reset_states()
        probabilities = []
        for start in range(0, len(wav), window_size_samples):
            chunk = wav[start:start + window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = self.torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
            probabilities.append(float(self.model(chunk, self.config.sampling_rate).item()))
        self.model.reset_states()
        return probabilities
