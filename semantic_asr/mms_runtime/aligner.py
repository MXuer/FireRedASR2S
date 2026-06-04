import logging

import torch
import torchaudio.functional as F

from semantic_asr.mms_runtime.align_utils import (
    get_spans,
    get_uroman_tokens,
    load_model_dict,
    merge_repeats,
    time_to_frame,
)
from semantic_asr.mms_runtime.model_registry import MMS_CODE_MAP

SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30

logger = logging.getLogger("semantic_asr.mms_runtime.aligner")


class MmsAligner:
    def __init__(self, model_path: str, device: str, uroman_path: str = "uroman/bin") -> None:
        logger.info("Loading MMS aligner model from %s", model_path)
        self.model, self.dictionary = load_model_dict(model_path)
        self.dictionary["<star>"] = len(self.dictionary)
        self.model.eval().to(device)
        self.uroman_path = uroman_path
        self.device = device

    def align(
        self,
        transcripts: list[str],
        waveform,
        sample_rate: int,
        names: list[str],
        use_star: bool,
        language: str,
        raw_transcripts: list[str],
    ) -> list[dict]:
        language = MMS_CODE_MAP.get(language, language)
        transcripts = [text for text in transcripts if text.strip()]
        norm_transcripts = [text.strip().lower() for text in transcripts]
        tokens = get_uroman_tokens(norm_transcripts, self.uroman_path, language)
        if use_star:
            stars = ["<star>"] * len(tokens)
            tokens = [item for pair in zip(tokens, stars) for item in pair]
            tokens = ["<star>"] + tokens
            transcripts = [item for pair in zip(transcripts, stars) for item in pair]
            transcripts = ["<star>"] + transcripts
            norm_transcripts = [item for pair in zip(norm_transcripts, stars) for item in pair]
            norm_transcripts = ["<star>"] + norm_transcripts

        segments, stride = self.get_alignments(waveform, sample_rate, tokens, use_star)
        spans = get_spans(tokens, segments)
        align_segments = []
        true_index = 0
        for i, transcript in enumerate(transcripts):
            if transcript == "<star>":
                continue
            span = spans[i]
            audio_start = round(span[0].start * stride / 1000, 3)
            audio_end = round(span[-1].end * stride / 1000, 3)
            align_segments.append({
                "start": round(audio_start, 3),
                "end": round(audio_end, 3),
                "duration": round(audio_end - audio_start, 3),
                "clean_text": transcript,
                "text": raw_transcripts[true_index],
                "name": names[true_index],
            })
            true_index += 1
        return align_segments

    def get_alignments(self, waveform, sample_rate: int, tokens: list[str], use_star: bool):
        emissions, stride = self.generate_emissions(waveform, sample_rate)
        time_steps, _ = emissions.size()
        if use_star:
            emissions = torch.cat([emissions, torch.zeros(time_steps, 1).to(self.device)], dim=1)

        token_indices = [self.dictionary[c] for c in " ".join(tokens).split(" ") if c in self.dictionary]
        blank = self.dictionary["<blank>"]
        # torchaudio forced_align has unstable CUDA behavior for some scripts.
        # Keep model emissions on GPU, but run the lightweight DP alignment on CPU.
        alignment_emissions = emissions.to("cpu")
        targets = torch.tensor(token_indices, dtype=torch.int32)
        input_lengths = torch.tensor(alignment_emissions.shape[0]).unsqueeze(-1)
        target_lengths = torch.tensor(targets.shape[0]).unsqueeze(-1)
        path, _ = F.forced_align(
            alignment_emissions.unsqueeze(0),
            targets.unsqueeze(0),
            input_lengths,
            target_lengths,
            blank=blank,
        )
        path = path.squeeze().to("cpu").tolist()
        segments = merge_repeats(path, {v: k for k, v in self.dictionary.items()})
        del alignment_emissions
        del targets
        del emissions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return segments, stride

    def generate_emissions(self, waveform, sample_rate: int):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T
        if sample_rate != SAMPLING_FREQ:
            waveform = F.resample(waveform, sample_rate, SAMPLING_FREQ)
            sample_rate = SAMPLING_FREQ
        waveform = waveform.to(self.device)
        total_duration = waveform.size(1) / sample_rate

        emissions_arr = []
        with torch.inference_mode():
            i = 0
            while i < total_duration:
                segment_start_time, segment_end_time = (i, i + EMISSION_INTERVAL)
                context = EMISSION_INTERVAL * 0.1
                input_start_time = max(segment_start_time - context, 0)
                input_end_time = min(segment_end_time + context, total_duration)
                waveform_split = waveform[
                    :,
                    int(sample_rate * input_start_time):int(sample_rate * input_end_time),
                ]

                model_outs, _ = self.model(waveform_split)
                emissions = model_outs[0]
                emission_start_frame = time_to_frame(segment_start_time)
                emission_end_frame = time_to_frame(segment_end_time)
                offset = time_to_frame(input_start_time)
                emissions = emissions[emission_start_frame - offset:emission_end_frame - offset, :]
                emissions_arr.append(emissions)
                i += EMISSION_INTERVAL

        emissions = torch.cat(emissions_arr, dim=0).squeeze()
        emissions = torch.log_softmax(emissions, dim=-1)
        stride = float(waveform.size(1) * 1000 / emissions.size(0) / sample_rate)
        del waveform
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return emissions, stride
