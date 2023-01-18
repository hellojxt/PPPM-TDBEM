import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set_theme(style="darkgrid")


def SNR(ground_truth, prediction):
    return 10*np.log(np.mean(ground_truth ** 2) / np.mean((ground_truth - prediction) ** 2))


def normalize(data):
    return data / (data**2).sum()**0.5


def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


root_dir = "../../dataset/static/"
sub_dirs = glob(root_dir + "*")
check_dir("./img/")
for sub_dir in sub_dirs:
    basename = sub_dir.split("/")[-1]
    output_dir = sub_dir + "/output/"
    img_dir = "./img/" + basename + "/"
    check_dir(img_dir)
    os.system("cp " + sub_dir + "/mesh.obj " + img_dir)
    sub_dirs = glob(output_dir + "0.*")
    sub_dirs = sorted(sub_dirs, key=lambda x: float(x.split("/")[-1]))
    grid_size_list = [float(x.split("/")[-1]) for x in sub_dirs]
    method_list = ['pppm', 'ghostcell1', 'ghostcell2', 'groundtruth']
    label_dict = {'pppm': 'PPPM', 'ghostcell1': 'Ghost Cell 1st',
                  'ghostcell2': 'Ghost Cell 2st', 'groundtruth': 'Ground Truth'}
    SNR_data = {'pppm': [], 'ghostcell1': [],
                'ghostcell2': [], 'groundtruth': []}
    time_data = {'pppm': [], 'ghostcell1': [],
                 'ghostcell2': [], 'groundtruth': []}
    for sub_dir, grid_size in zip(sub_dirs, grid_size_list):
        method_data = {}
        for method in method_list:
            method_data[method] = np.zeros(0)
            ffat_file_list = glob(sub_dir + "/" + method + "/ffat*.txt")
            for ffat_file in ffat_file_list:
                data = np.loadtxt(ffat_file).reshape(-1)
                method_data[method] = np.append(method_data[method], data)
                if method != 'groundtruth':
                    method_data[method] = method_data[method]
            time = float(np.loadtxt(sub_dir + "/" + method + "/cost_time.txt"))
            time_data[method].append(time)
        gt = method_data['groundtruth']
        for method in method_list:
            if method == 'groundtruth':
                continue
            SNR_data[method].append(SNR(gt, method_data[method]))
    plt.figure(figsize=(15, 8))
    for method in method_list:
        if method == 'groundtruth':
            continue
        plt.plot(grid_size_list, SNR_data[method],
                 label=label_dict[method], marker='o')
    plt.legend()
    SMALL_SIZE = 8
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.xlabel("Grid Size")
    plt.ylabel("SNR (dB)")
    plt.savefig(img_dir + "SNR.png", dpi=300,
                bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(15, 8))
    for method in method_list:
        if method == 'groundtruth':
            continue
        plt.plot(grid_size_list, time_data[method],
                 label=label_dict[method], marker='o')
    plt.legend()
    SMALL_SIZE = 8
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.xlabel("Grid Size")
    plt.ylabel("Time (s)")
    plt.savefig(img_dir + "time.png", dpi=300,
                bbox_inches='tight', pad_inches=0)
