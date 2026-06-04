import os
import torch
import torchaudio
import re
import sox
import json
import chardet
import argparse


from l2s.align.text_normalization import text_normalize
from l2s.align.align_utils import (
    get_uroman_tokens,
    time_to_frame,
    load_model_dict,
    merge_repeats,
    get_spans,
)
import torchaudio.functional as F

SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30


def generate_emissions(model, audio_file, device):
    waveform, _ = torchaudio.load(audio_file, format='wav')  # waveform: channels X T
    waveform = waveform.to(device)
    total_duration = sox.file_info.duration(audio_file)

    audio_sf = sox.file_info.sample_rate(audio_file)
    assert audio_sf == SAMPLING_FREQ

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
                int(SAMPLING_FREQ * input_start_time) : int(
                    SAMPLING_FREQ * (input_end_time)
                ),
            ]

            model_outs, _ = model(waveform_split)
            emissions_ = model_outs[0]
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)

            emissions_ = emissions_[
                emission_start_frame - offset : emission_end_frame - offset, :
            ]
            emissions_arr.append(emissions_)
            i += EMISSION_INTERVAL

    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)

    stride = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)
    
    del waveform
    torch.cuda.empty_cache()

    return emissions, stride


def get_alignments(
    audio_file,
    tokens,
    model,
    dictionary,
    use_star,
    device
):
    # Generate emissions
    emissions, stride = generate_emissions(model, audio_file, device)
    T, N = emissions.size()
    if use_star:
        emissions = torch.cat([emissions, torch.zeros(T, 1).to(device)], dim=1)

    # Force Alignment
    if tokens:
        token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
    else:
        print(f"Empty transcript!!!!! for audio file {audio_file}")
        token_indices = []

    blank = dictionary["<blank>"]
    
    targets = torch.tensor(token_indices, dtype=torch.int32).to(device)
    input_lengths = torch.tensor(emissions.shape[0]).unsqueeze(-1)
    target_lengths = torch.tensor(targets.shape[0]).unsqueeze(-1)

    path, _ = F.forced_align(
        emissions.unsqueeze(0), targets.unsqueeze(0), input_lengths, target_lengths, blank=blank
    )
    path = path.squeeze().to("cpu").tolist()
    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    del targets
    del emissions
    torch.cuda.empty_cache()
    return segments, stride

