import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import librosa

SAMPLE_RATE = 8000 #音频文件采样率
N_FFT = 128 #fft的N

def E_frame(f):#计算能量谱
    mag_f = np.absolute(f)
    pow_f = (1.0/N_FFT) * ((mag_f)**2)
    return pow_f 

def main():
    dataset = np.load("data_fft_with_type.npz")
    fft_m = dataset['matrix1'] #fft result matrix
    type_m = dataset['matrix2'] #type matrix

    #初始化mel滤波器参数
    n_mel = 120 #mel过滤器的数量
    low_freq = 0#频率下限
    high_freq = 2595* np.log10(1+SAMPLE_RATE/700)# 频率上限
    mel_points = np.linspace(low_freq, high_freq, n_mel + 2)#中心频率
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bins = np.floor((N_FFT + 1 )* hz_points/ SAMPLE_RATE)
    #构建滤波器
    fbank = np.zeros((n_mel, int(np.floor(N_FFT))))
    for i in range(n_mel ):
        left = int(bins[i - 1])
        center = int(bins[i])
        right = int(bins[i + 1])
        for j in range(left, center):
            fbank[i - 1, j] = (j - bins[i - 1]) / (bins[i] - bins[i - 1])
        for j in range(center, right):
            fbank[i - 1, j] = (bins[i + 1] - j) / (bins[i + 1] - bins[i])

    mfcc_m = []
    for i in range(fft_m.shape[0]):
        #计算能量谱
        e_frame = E_frame(fft_m[i])
        #梅尔滤波能量
        fbank_feat = np.dot(e_frame,fbank.T)
        fbank_feat = np.where(fbank_feat == 0, np.finfo(float).eps, fbank_feat) #防止0
        fbank_feat = 20 * np.log10(fbank_feat)                                

        # 计算mfcc系数
        mfcc_feat = dct(fbank_feat, type=2, axis=1, norm='ortho')[:, :12]
        mfcc_m.append(mfcc_feat)
        print(mfcc_feat)
        #librosa.display.specshow(data=mfcc_feat,
        #                     sr=SAMPLE_RATE,
        #                    n_fft=N_FFT,
        #                     hop_length=SAMPLE_RATE // 100,
        #                     win_length=SAMPLE_RATE // 40,
        #                     x_axis="s")
        
        #plt.colorbar(format="%d")
        #plt.show()

    
    np.savez('mfcc12_result_with_type.npz',matrix1=mfcc_m,matrix2=type_m)




if __name__ == '__main__':
    main()
