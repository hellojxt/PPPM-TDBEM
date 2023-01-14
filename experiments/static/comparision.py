import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.io import wavfile

def plot_data(datas, names, title = None, labels = None, clip_length = 256):
    xs = np.arange(min([len(data) for data in datas] + [clip_length]))
    plt.figure(figsize=(10, 5))
    for data, name in zip(datas, names):
        plt.plot(xs, data[:clip_length], label=name)
    if labels is not None:
        plt.xticks(xs, labels[:clip_length])
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()

def SNR(ground_truth, prediction):
    return np.mean(ground_truth ** 2) / np.mean((ground_truth - prediction) ** 2)

def multi_SNR(ground_truth, prediction, point_num, clip_idx, mode='logsnr_avg'):
    cnt_snr = 0
    len_signal = len(ground_truth) // point_num
    for i in range(point_num):
        cnt_gt = ground_truth[len_signal * i + clip_idx: len_signal * (i + 1)]
        cnt_pred = prediction[len_signal * i + clip_idx: len_signal * (i + 1)]
        if(mode == 'logsnr_avg'):
            cnt_snr += 10 * np.log10(SNR(cnt_gt, cnt_pred))
        elif(mode == 'snr_avg'):
            cnt_snr += SNR(cnt_gt, cnt_pred)

    # now multi = avg SNR
    # 对SNR取平均是否合理？
    multi_snr = cnt_snr / point_num
    return multi_snr

def one_multi_SNR(ground_truth, prediction, clip_idx, index, len_signal):
    cnt_gt = ground_truth[len_signal * index + clip_idx: len_signal * (index + 1)]
    cnt_pred = prediction[len_signal * index + clip_idx: len_signal * (index + 1)]
    cnt_snr = 10 * np.log10(SNR(cnt_gt, cnt_pred))
    return cnt_snr

# 保存波形到wav
def save_wav(data, filename, sample_rate=44100):
    data = data.astype(np.float32)
    # data = data / np.max(np.abs(data))
    data = data * 32767
    data = data.astype(np.int16)
    wavfile.write(filename, sample_rate, data)


model_name = "sphere"
dirs = glob('experiments/static/output/' + model_name + '.obj/*')
dirs = sorted(dirs, key=lambda x: x.split('/')[-1])
subdirs = []
for dir in dirs:
    # if dir[-4:] != ".png":
    if '6.0' in dir:
        subdirs.append(sorted(glob(dir + '/0.*'), key=lambda x: x.split('/')[-1]))
keys = [dir.split('/')[-1] for dir in subdirs[0]]

# 去除非以网格大小命名的文件夹的数据
# i = 0
# while i < len(keys):
#     if keys[i][-4:] == ".png":
#         del keys[i]
#     else:
#         i += 1

pppm_data = [[] for i in range(len(keys))]
ghost_1st_data = [[] for i in range(len(keys))]
ghost_2nd_data = [[] for i in range(len(keys))]
pppm_precompute_times = [[] for i in range(len(keys))]
pppm_times = [[] for i in range(len(keys))]
ghost_1st_times = [[] for i in range(len(keys))]
ghost_2nd_times = [[] for i in range(len(keys))]

# multi情形clip移到SNR计算中
clip_idx = 256
point_num = 156
bb_point_num = 26
bb_num = 6

# 用来给多个点每个点输出一个结果
# sample_points = 6
# sample_lists = [0, 1, 2, 6, 7, 12] # 根据对称性，对球体来说只有六种不同的采样点

# 这里计算每一个包围盒的所有点的平均SNR
pppm_data_multi = [([[] for i in range(len(keys))]) for i in range(bb_num)]
ghost_1st_data_multi = [([[] for i in range(len(keys))]) for i in range(bb_num)]
ghost_2nd_data_multi = [([[] for i in range(len(keys))]) for i in range(bb_num)]

for dir_parent in subdirs:
    for i, dir in enumerate(dir_parent):
        # analytical = np.loadtxt(dir + '/analytical_solution_multi.txt')[clip_idx:]
        # pppm = np.loadtxt(dir + '/pppm_solution_multi.txt')[clip_idx:]
        # ghost_cell_1st = np.loadtxt(dir + '/ghost_cell_1st_multi.txt')[clip_idx:]
        # ghost_cell_2nd = np.loadtxt(dir + '/ghost_cell_2nd_multi.txt')[clip_idx:]
        analytical = np.loadtxt(dir + '/analytical_solution_multi.txt')[:]
        pppm = np.loadtxt(dir + '/pppm_solution_multi.txt')[:]
        ghost_cell_1st = np.loadtxt(dir + '/ghost_cell_1st_multi.txt')[:]
        ghost_cell_2nd = np.loadtxt(dir + '/ghost_cell_2nd_multi.txt')[:]
        with open(dir + '/cost_time.txt') as f:
            cost_times = f.readlines()
            pppm_precompute_times[i].append(float(cost_times[0].split(' ')[-1]))
            pppm_times[i].append(float(cost_times[1].split('=')[1]))
            ghost_1st_times[i].append(float(cost_times[2].split('=')[1]))
            ghost_2nd_times[i].append(float(cost_times[3].split('=')[1]))

        # 需要在这里处理多点采样问题
        # 目前对多点采样的SNR取平均计算
        pppm_data[i].append(multi_SNR(analytical, pppm, point_num, clip_idx, 'logsnr_avg'))
        ghost_1st_data[i].append(multi_SNR(analytical, ghost_cell_1st, point_num, clip_idx, 'logsnr_avg'))
        ghost_2nd_data[i].append(multi_SNR(analytical, ghost_cell_2nd, point_num, clip_idx, 'logsnr_avg'))

        # pppm_data[i].append(SNR(analytical, pppm))
        # ghost_1st_data[i].append(SNR(analytical, ghost_cell_1st))
        # ghost_2nd_data[i].append(SNR(analytical, ghost_cell_2nd))

        # 依次分别计算多点输入j
        step_bb = len(analytical) // point_num * bb_point_num
        for j in range(bb_num):
            pppm_data_multi[j][i].append( \
                multi_SNR(analytical[j * step_bb:(j+1) * step_bb], pppm[j * step_bb:(j+1) * step_bb], bb_point_num, clip_idx, 'logsnr_avg'))
            ghost_1st_data_multi[j][i].append( \
                multi_SNR(analytical[j * step_bb:(j+1) * step_bb], ghost_cell_1st[j * step_bb:(j+1) * step_bb], bb_point_num, clip_idx, 'logsnr_avg'))
            ghost_2nd_data_multi[j][i].append( \
                multi_SNR(analytical[j * step_bb:(j+1) * step_bb], ghost_cell_2nd[j * step_bb:(j+1) * step_bb], bb_point_num, clip_idx, 'logsnr_avg'))


        # 保存cnt_point=0的wav结果
        # save_wav_path = dir + "/"
        # save_wav(analytical[clip_idx: len(analytical) // point_num], save_wav_path + "analytical.wav")
        # save_wav(pppm[clip_idx: len(pppm) // point_num], save_wav_path + "pppm.wav")
        # save_wav(ghost_cell_1st[clip_idx: len(ghost_cell_1st) // point_num], save_wav_path + "ghost_cell_1st.wav")
        # save_wav(ghost_cell_2nd[clip_idx: len(ghost_cell_2nd) // point_num], save_wav_path + "ghost_cell_2nd.wav")


# pppm_data = 10 * np.log10(np.array(pppm_data)).mean(axis=1)
# ghost_1st_data = 10 * np.log10(np.array(ghost_1st_data)).mean(axis=1)
# ghost_2nd_data = 10 * np.log10(np.array(ghost_2nd_data)).mean(axis=1)
pppm_data = np.array(pppm_data).mean(axis=1)
ghost_1st_data = np.array(ghost_1st_data).mean(axis=1)
ghost_2nd_data = np.array(ghost_2nd_data).mean(axis=1)

pppm_precompute_times = np.array(pppm_precompute_times).mean(axis=1)
pppm_times = np.array(pppm_times).mean(axis=1)
ghost_1st_times = np.array(ghost_1st_times).mean(axis=1)
ghost_2nd_times = np.array(ghost_2nd_times).mean(axis=1)

# 逐点保存采样
for i in range(bb_num):
    plot_data([pppm_data_multi[i], ghost_1st_data_multi[i], ghost_2nd_data_multi[i]], \
        ['PPPM', 'Ghost Cell 1st', 'Ghost Cell 2nd'], labels=keys, title="average SNR")
    path = 'experiments/static/output/' + model_name + ".obj/6.0/"
    plt.savefig(path + "SNR_bbsize_ " + str(0.05 * (3 + i))[:4] + ".png") 

plot_data([pppm_data, ghost_1st_data, ghost_2nd_data], ['PPPM', 'Ghost Cell 1st', 'Ghost Cell 2nd'], labels=keys, title="average SNR")
path = 'experiments/static/output/' + model_name + ".obj/6.0/"
plt.savefig(path + "multi_SNR.png") 

plot_data([pppm_times, pppm_times + pppm_precompute_times, ghost_1st_times, ghost_2nd_times], 
    ['PPPM', 'PPPM (with precomputation)', 'Ghost Cell 1st', 'Ghost Cell 2nd'], labels=keys, title="average time")
plt.savefig(path + "time.png") 


