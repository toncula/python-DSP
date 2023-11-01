import wave
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import wiener
def frame_and_window(wav_filename, frame_time, frame_shift_percent, window_type='hamming'):
    # 打开 WAV 文件
    with wave.open(wav_filename, 'rb') as wf:
        num_frames = wf.getnframes()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()

        frames = wf.readframes(-1)  # 读取所有帧

    # 将字节数据转换为 NumPy 数组
    frames_np = np.frombuffer(frames, dtype=np.int16)
    #frames_np=wiener(frames_np)
    if np.max(frames_np)<300:
        return 'error',0
    frame_size=int(frame_time*sample_rate)
    frame_shift=int(frame_shift_percent*frame_size) # 计算窗函数
    if window_type == 'hamming':
        window = scipy.signal.windows.hamming(frame_size)
    elif window_type == 'hanning':
        window = scipy.signal.windows.hann(frame_size)
    elif window_type == 'blackman':
        window = scipy.signal.windows.blackman(frame_size)
    else:
        window = np.ones(frame_size)

    # 计算分帧数
    num_frames = (len(frames_np) - frame_size) // frame_shift + 1

    # 分帧加窗
    framed_data = np.zeros((num_frames, frame_size), dtype=np.int16)

    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_size
        frame = frames_np[start:end]
        framed_data[i, :] = frame * window

    return frames_np,framed_data

def frame_feature(framed_data):
    k,l=framed_data.shape
    energy=(framed_data/100)**2
    data_energy=np.sum(energy,axis=1)
    top_five_max = np.partition(data_energy, -5)[-5:]
    top_mean = np.mean(top_five_max)*0.01
    data_energy=data_energy/top_mean
    signs = np.sign(framed_data)
    diff = np.diff(signs,axis=1)
    zero_crossings = np.sum(diff > 0,axis=1)
    return data_energy,zero_crossings,top_mean,l
def vad(data_en,zero_cro,l,th1_rate,th2_rate,zro):
    index = np.argsort(data_en)
    potential_tuple=(0,0,0)
    while data_en[index[-1]]>th1_rate:
        max_index=index[-1]
        n1,n2=max_index,max_index
        for j in range(max_index,0,-1):
            if data_en[j]<th1_rate:
                n1=j
                break
        for k in range(max_index,data_en.shape[0]):
            if data_en[k]<th1_rate:
                n2=k
                break
        n3,n4=n1,n2
        for m in range(n1,0,-1):
            if (data_en[m]>th2_rate and (n1-m)<(n2-n1)) :
                n3=n3-1
            else:break
        for i in range(n2,data_en.shape[0]):
            if (data_en[i]>th2_rate and (i-n2)<(n2-n1))  :
                n4=n4+1
            else:break

        if n4-n3>potential_tuple[0]:
            potential_tuple=(n4-n3,n3,n4)
        condition = (n3 > index) | (index > n4)
        index = index[condition]
        if index.shape[0]==0:
            break
    n3,n4=potential_tuple[1],potential_tuple[2]
    n5, n6 = n3, n4
    for i in range(n3,-1,-1):
        if zero_cro[i]<zro and (n3-i)<(n4-n3)/4:
            n5=i
            break
    for i in range(n4,data_en.shape[0]):
        if zero_cro[i] < zro and (i-n4)<(n4-n3)/4:
            n6 = i
            break
    true_data_en=data_en[n5:n6]
    true_zero_cro=zero_cro[n5:n6]

    return true_data_en,true_zero_cro,n5,n6
#特征：清音（开头）长度占比，浊音（中间）占比，清音（开头）过零率求和占比，浊音（中间）过零率求和占比,短时平均幅度差
def feature_engineering(data_energy,data_zerocross,data_frame,start,end):
    x = data_frame.shape[1]
    y = end - start
    aver_energy=data_energy*100/np.sum(data_energy)
    aver_zerocross=data_zerocross*100/np.sum(data_zerocross)
    mag_diff=np.sum(np.abs(data_frame[start:end,0:x-1]-data_frame[start:end,1:x]),axis=1)
    aver_mag_diff=mag_diff*100/np.sum(mag_diff)
    x_original = np.linspace(0, 1, len(aver_energy))
    f1 = interp1d(x_original, aver_energy)
    f2 = interp1d(x_original, aver_mag_diff)
    f3 = interp1d(x_original, aver_zerocross)

    # 创建新的自变量序列，这里假设你想要的新数据长度为50
    x_new = np.linspace(0, 1, 50)

    # 使用插值函数得到新的数组
    aver_energy = f1(x_new)
    aver_mag_diff = f2(x_new)
    aver_zerocross = f3(x_new)
    gate=0.01
    energy_zerocross=aver_energy
    list=np.where(energy_zerocross>gate)
    if len(list[0])>0:
        id1=list[0][0]
    else:id1=0
    if len(list[0])>1:
        id2=list[0][-1]+1
    else:id2=y
    if id2-id1<14:
        return 'error3',0,0

    # plt.subplot(1,3,1)
    # plt.plot(aver_energy)
    # plt.scatter([id1,id2-1],[aver_energy[id1],aver_energy[id2-1]],color='yellow')
    # plt.subplot(1, 3, 2)
    # plt.plot(aver_zerocross)
    # plt.scatter([id1, id2 - 1], [aver_zerocross[id1], aver_zerocross[id2 - 1]], color='yellow')
    # plt.subplot(1, 3, 3)
    # plt.plot(aver_mag_diff)
    # plt.scatter([id1, id2 - 1], [aver_mag_diff[id1], aver_mag_diff[id2 - 1]], color='yellow')
    #
    # plt.show()

    dull_ratio=(id2-id1)/y*10
    clean_ratio=id1/y*10
    dull_zero_sum=np.sum(aver_zerocross[id1:id2])/10
    clean_zero_sum=np.sum(aver_zerocross[0:id1])/10
    part_zerocross=np.zeros(20)
    part_energy=np.zeros(20)
    part_mag_diff=np.zeros(20)
    part_lenth=int((id2-id1-4)//20)
    part_start=int((id2-id1-part_lenth*20)//2)
    kernel=np.array([0,0,15,0,0])/15
    for i in range(20):
        part_energy[i]=np.sum(aver_energy[part_start+i*part_lenth-2:part_start+i*part_lenth+3]*kernel)
        part_mag_diff[i]=np.sum(aver_mag_diff[part_start+i*part_lenth-2:part_start+i*part_lenth+3]*kernel)
        part_zerocross[i]=np.sum(aver_zerocross[part_start+i*part_lenth-2:part_start+i*part_lenth+3]*kernel)
    feature=np.hstack((np.array([dull_ratio,clean_ratio,dull_zero_sum,clean_zero_sum]),
                       part_zerocross,part_energy,part_mag_diff))
    return feature,id1,id2

def feature_get(filename,frame_time,frame_shift_percent,window_type):
    frames_np, framed_data = frame_and_window(filename, frame_time, frame_shift_percent, window_type)
    if isinstance(frames_np, str):
        return 'error1',0
    data_en, zero_cro, mean, l = frame_feature(framed_data)
    data_en_t, zero_cro_t, start, end = vad(data_en, zero_cro, l, 10, 0.05, 10)
    print(end-start)
    if end-start<10:
        return 'error2',0
    data_feature, id1, id2 = feature_engineering(data_en_t, zero_cro_t, framed_data, start, end)
    if isinstance(data_feature, str):
        return 'error3',0
    parts = filename.split('_')
    audio_category = ''
    if len(parts) >= 3:
        audio_category = int(parts[2])
    return data_feature,audio_category
def main():
    wav_filename = 'dsp_experiment1/58_7_3.wav'
    frame_time = 0.032  # 帧大小
    frame_shift_percent = 0.5  # 帧移
    window_type = 'hamming'
    frames, framed_data = frame_and_window(wav_filename, frame_time, frame_shift_percent, window_type)
    if isinstance(frames, str):
        return 'error'
    data_en, zero_cro, mean, l = frame_feature(framed_data)
    data_en_t, zero_cro_t, start, end = vad(data_en, zero_cro, l, 10, 0.05, 10)
    print(end-start)
    data_feature, id1, id2 = feature_engineering(data_en_t, zero_cro_t, framed_data, start, end)
    s = int(start * frame_time * frame_shift_percent * 8000)
    e = int(end * frame_time * frame_shift_percent * 8000)

    plt.subplot(2, 2, 1)
    plt.plot(data_en)
    plt.scatter([start, end], [data_en[start], data_en[end]], color='red')
    plt.subplot(2, 2, 2)
    plt.plot(data_en_t)
    #plt.scatter([id1, id2-1], [data_en_t[id1], data_en_t[id2-1]], color='green')
    plt.subplot(2, 2, 3)
    plt.plot(frames)
    plt.scatter([s, e], [frames[s], frames[e]], c='red')
    plt.subplot(2, 2, 4)
    plt.plot(zero_cro_t)
    #plt.scatter([id1, id2 - 1], [zero_cro_t[id1], zero_cro_t[id2 - 1]], color='green')
    plt.show()
    from scipy.io import wavfile
    wavfile.write("output.wav", 8000, frames[s:e])

if __name__=="__main__":
    main()

