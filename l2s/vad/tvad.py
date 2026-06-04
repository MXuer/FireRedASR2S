import re
import numpy as np
from ten_vad import TenVad
import scipy.io.wavfile as Wavfile
from itertools import combinations


def get_speech_timestamps(
    probs,
    sampling_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float('inf'),
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 256, # ten-vad 默认一帧的大小 16ms
    speech_pad_ms: int = 0,
):
    """
    基于 silero-vad 逻辑的 VAD 后处理，生成 ten-vad 的时间戳。
    
    参数:
        probs: ten-vad 输出的每一帧是语音的概率列表/数组
        sampling_rate: 采样率
        threshold: 判定为语音的概率阈值
        min_speech_duration_ms: 最短语音长度，小于此长度的片段将被丢弃
        min_silence_duration_ms: 最短静音长度，小于此长度的停顿将合并前后语音段
        window_size_samples: 每一帧对应的采样点数 (ten-vad 默认通常是 512 或 1024)
        speech_pad_ms: 语音前后的扩充长度
    """
    
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - speech_pad_samples * 2

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15 # 类似于 silero 的迟滞阈值
    
    # 转换为采样点位置
    temp_end = 0 

    for i, prob in enumerate(probs):
        # 当前帧对应的采样点起始位置
        current_sample = i * window_size_samples

        # 1. 触发语音开始
        if (prob >= threshold) and temp_end:
            temp_end = 0 # 重置静音计时

        if (prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = current_sample
            continue

        # 2. 触发语音结束 (低于阈值)
        if (prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = current_sample
            
            # 只有当静音持续时间超过 min_silence_samples 时才真正结束
            if current_sample - temp_end >= min_silence_samples:
                current_speech['end'] = temp_end
                
                # 验证语音长度是否达标
                if (current_speech['end'] - current_speech['start']) >= min_speech_samples:
                    speeches.append(current_speech)
                
                current_speech = {}
                triggered = False
                temp_end = 0
            continue

    # 3. 处理最后一段
    if triggered and current_speech:
        current_speech['end'] = len(probs) * window_size_samples
        if (current_speech['end'] - current_speech['start']) >= min_speech_samples:
            speeches.append(current_speech)

    # 4. 加上 padding 并转换为秒或毫秒
    for i, speech in enumerate(speeches):
        speech['start'] = max(0, speech['start'] - speech_pad_samples)
        speech['end'] = speech['end'] + speech_pad_samples
        
        # 转换为秒 (便于阅读)
        speech['start_sec'] = round(speech['start'] / sampling_rate, 3)
        speech['end_sec'] = round(speech['end'] / sampling_rate, 3)

    return speeches

def do_ten_vad(wav_file):
    sr, data = Wavfile.read(wav_file)
    hop_size = 256  # 16 ms per frame
    threshold = 0.5
    ten_vad_instance = TenVad(hop_size, threshold)  # Create a TenVad instance
    num_frames = data.shape[0] // hop_size
    out_probabilities = []
    for i in range(num_frames):
        audio_data = data[i * hop_size: (i + 1) * hop_size]
        out_probability, out_flag = ten_vad_instance.process(audio_data) #  Out_flag is speech indicator (0 for non-speech signal, 1 for speech signal)
        out_probabilities.append(out_probability)

    
    timestamps = get_speech_timestamps(
        probs=out_probabilities,
        sampling_rate=16000,
        window_size_samples=256,
        threshold=0.5,
        min_silence_duration_ms=100
    )
    return timestamps




# 使用示例 (伪代码)
if __name__ == "__main__":
    timestamps = do_ten_vad('/data_151/duhu/DBC/ASR/阿里方言跑识别筛选数据/第八批/ds_20260410/qingdao/audio/htrs_spk0116_qingdao_0488078.wav')
    print(timestamps)
    