import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from config import get_config

config = get_config()   # 获取参数

# 音频数据集的路径
audio_path= 'D:/file/UCAS_Postgraduate_Grade_1/deep_learning/4project/1_project_dw/实验7-声纹识别/DS_10283_2651/VCTK_Corpus/wav48'     # utterance datasets

def save_spectrogram_tisv():
    """ 文本独立话语的预处理. 
        将说话者音频的log-mel-spectrogram保存为numpy文件.
        利用DB语音检测对每个话语进行分割，保存每个话语的前180帧和后180帧。
        输入 : 说话数据集(VTCK)
        输出：每个说话者的log-mel-spectrogram
    """
    print("start text independent utterance feature extraction")
    os.makedirs(config.train_path, exist_ok=True)   # 创建文件夹保存训练文件
    os.makedirs(config.test_path, exist_ok=True)    # 创建文件夹保存测试文件

    utter_min_len = (config.tisv_frame * config.hop + config.window) * config.sr    # 话语长度的下界
    total_speaker_num = len(os.listdir(audio_path))
    train_speaker_num= (total_speaker_num//10)*9            # 分割数据， 90% train ， 10% test
    print("total speaker number : %d"%total_speaker_num)
    print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))
    for i, folder in enumerate(os.listdir(audio_path)):
        speaker_path = os.path.join(audio_path, folder)     # 每个说话者的路径
        print("%dth speaker processing..."%i)
        utterances_spec = []
        k=0
        for utter_name in os.listdir(speaker_path):
            utter_path = os.path.join(speaker_path, utter_name)         # 每个话语的路径
            utter, sr = librosa.core.load(utter_path, config.sr)        # 加载话语音频
            intervals = librosa.effects.split(utter, top_db=20)         # 有音区域检测
            for interval in intervals:
                if (interval[1]-interval[0]) > utter_min_len:           # 如果话语长度太长
                    utter_part = utter[interval[0]:interval[1]]         # 保存前180帧和后180帧
                    S = librosa.core.stft(y=utter_part, n_fft=config.nfft,
                                          win_length=int(config.window * sr), hop_length=int(config.hop * sr))
                    S = np.abs(S) ** 2
                    mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
                    S = np.log10(np.dot(mel_basis, S) + 1e-6)           # 话语的log mel spectrogram

                    utterances_spec.append(S[:, :config.tisv_frame])    # 前180帧
                    utterances_spec.append(S[:, -config.tisv_frame:])   # 后180帧

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        if i<train_speaker_num:      # 将谱图保存为numpy文件
            np.save(os.path.join(config.train_path, "speaker%d.npy"%i), utterances_spec)
        else:
            np.save(os.path.join(config.test_path, "speaker%d.npy"%(i-train_speaker_num)), utterances_spec)


if __name__ == "__main__":
    save_spectrogram_tisv()
