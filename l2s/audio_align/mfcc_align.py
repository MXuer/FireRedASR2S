import librosa
import numpy as np
from scipy.signal import correlate
from tqdm import tqdm
def align_audio(long_path, short_path, sr=16000, anchor_duration=60):
    # 1. 加载音频并统一采样率（归一化为单声道）
    print("Loading audio files...")
    # 只加载短音频的前 30s 作为锚点
    y_short, _ = librosa.load(short_path, sr=sr, mono=True, duration=anchor_duration)
    # 加载长音频全文
    y_long, _ = librosa.load(long_path, sr=sr, mono=True)

    # 2. 提取 MFCC 特征
    # 10ms 帧移 (160 samples at 16kHz)
    hop_length = 160
    print(f"Extracting MFCC features (hop_length={hop_length})...")
    mfcc_long = librosa.feature.mfcc(y=y_long, sr=sr, n_mfcc=13, hop_length=hop_length)
    mfcc_short = librosa.feature.mfcc(y=y_short, sr=sr, n_mfcc=13, hop_length=hop_length)

    # 3. 特征标准化 (CMVN: Cepstral Mean and Variance Normalization)
    print("Applying CMVN normalization...")
    def cmvn(mfcc):
        # mfcc shape: (n_mfcc, n_frames)
        # 对每一个维度（系数）在时间轴上做归一化，消除设备频响差异
        mu = np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True) + 1e-9
        return (mfcc - mu) / std

    # 转置为 (n_frames, n_mfcc) 方便按帧遍历
    mfcc_long = cmvn(mfcc_long).T
    mfcc_short = cmvn(mfcc_short).T

    # 4. 计算相似度
    print("Finding best match position by frame-wise Cosine Similarity mean...")
    
    len_long = len(mfcc_long)
    len_short = len(mfcc_short)
    
    max_avg_sim = -1.0
    best_frame = 0
    
    # 预先计算短音频每一帧的范数，并转换为单位向量
    mfcc_short_norm = np.linalg.norm(mfcc_short, axis=1, keepdims=True) + 1e-9
    mfcc_short_unit = mfcc_short / mfcc_short_norm

    # 为了加速，预先计算长音频所有帧的范数
    mfcc_long_norms = np.linalg.norm(mfcc_long, axis=1, keepdims=True) + 1e-9

    for i in tqdm(range(len_long - len_short + 1)):
        # 截取长音频的一段特征及其范数
        chunk = mfcc_long[i : i + len_short]
        chunk_norms = mfcc_long_norms[i : i + len_short]
        
        # 将当前 chunk 的每一帧归一化为单位向量
        chunk_unit = chunk / chunk_norms
        
        # 计算每一帧的余弦相似度 (单位向量的点积)
        frame_cos_sim = np.sum(chunk_unit * mfcc_short_unit, axis=1)
        
        # 求当前 30s 窗口内所有帧余弦相似度的均值
        avg_sim = np.mean(frame_cos_sim)
        
        if avg_sim > max_avg_sim:
            max_avg_sim = avg_sim
            best_frame = i

    # 5. 结果转换
    start_time = (best_frame * hop_length) / sr
    end_time = ((best_frame + len_short) * hop_length) / sr

    print(f"\nMatch Found!")
    print(f"Best Average Similarity: {max_avg_sim:.4f}")
    print(f"Start Time in Long Audio: {start_time:.3f} seconds")
    print(f"End Time in Long Audio: {end_time:.3f} seconds")
    print(f"Start Frame Index: {best_frame}")
    
    return best_frame, start_time
