import matplotlib.pyplot as plt

import voice_feature as vf
import os
import numpy as np
#改一下地址
DATA_FILE_HEAD=os.getcwd()
DATA_FILE=os.path.join(DATA_FILE_HEAD,'task2/dsp_experiment1')
print(DATA_FILE)
DATA_LIST=os.listdir(DATA_FILE)
#设置帧长，帧移，和窗函数
FRAMED_TIME=0.032
FRAMED_SHIFT=0.5
WINDOW_TYPE='hanming'

data_fft=[]
data_type=[]

for filename in DATA_LIST:
    filename=os.path.join(DATA_FILE,filename)
    frames_np, framed_data = vf.frame_and_window(filename, FRAMED_TIME, FRAMED_SHIFT, WINDOW_TYPE)
    print(filename)
    #error1是音频最高能量不正常，判断为没有录入有效声音
    if isinstance(frames_np, str):
        print('error1')
        continue
    data_en, zero_cro, mean, l = vf.frame_feature(framed_data)
    data_en_t, zero_cro_t, start, end = vf.vad(data_en, zero_cro, l, 10, 0.05, 10)
    #用双门限得到有效音频帧后，把所有的音频扩充或者裁剪到80帧
    if end - start < 10:
        #认为太短的音频无效
        print('error2')
        continue
    if end-start<80:
        framed_data_need = framed_data[start:end] # type: ignore
        framed_data_fft = np.fft.fft(framed_data_need, axis=-1)
        framed_data_fft_shifted = np.fft.fftshift(framed_data_fft, axes=-1)
        num_points = framed_data_fft_shifted.shape[1]
        positive_frequencies = framed_data_fft_shifted[:, num_points // 2:]
        pad=((0, 80 - (end-start)), (0, 0))
        positive_frequencies = np.pad(positive_frequencies, pad, mode='constant')
    else:
        decrease=end-start-80
        for i in range(decrease):
            if data_en[start+1]<1:
                start=start+1
            else:
                end=end-1
        framed_data_need=framed_data[start:end] # type: ignore
        framed_data_fft=np.fft.fft(framed_data_need,axis=-1)
        framed_data_fft_shifted = np.fft.fftshift(framed_data_fft, axes=-1)
        num_points = framed_data_fft_shifted.shape[1]
        positive_frequencies = framed_data_fft_shifted[:, num_points // 2:]

    data_fft.append(positive_frequencies)
    parts = filename.split('_')
    audio_category = ""
    if len(parts) >= 3:
        audio_category = int(parts[2])
    data_type.append(audio_category)

data_fft=np.array(data_fft)
data_type=np.array(data_type).reshape(len(data_type),1)
print(data_fft.shape)
print(data_type.shape)
#np.savez('data_fft_with_type.npz', matrix1=data_fft, matrix2=data_type)
