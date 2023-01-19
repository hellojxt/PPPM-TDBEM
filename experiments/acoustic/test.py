
import sys
sys.path.append('../../python_scripts')
import warnings
import meshio
from bem import BEMModel, bempp
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set_theme(style="darkgrid")
bempp.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "gpu"
bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "gpu"

obj_name = sys.argv[1]


def SNR(ground_truth, prediction):
    return 10 * np.log(np.mean(ground_truth ** 2) / np.mean((ground_truth - prediction) ** 2))


def process_acoustic_data(dir_name):
    names = dir_name.split("/")
    margin = 0.1

    def norm_ffat(ffat, dim):
        ffat = ffat.reshape((6, dim, dim))
        dim_with_margin = int(dim * (1 + margin))
        ffat_with_margin = np.zeros((6, dim_with_margin, dim_with_margin))
        for i in range(6):
            ffat[i] = ffat[i] / (ffat[i]**2).sum()**0.5
            ffat_with_margin[i, int(dim * margin / 2):int(dim * margin / 2) +
                             dim, int(dim * margin / 2):int(dim * margin / 2) + dim] = ffat[i]
        return ffat_with_margin.reshape((6 * dim_with_margin, dim_with_margin))
    pixel_pos = np.loadtxt(dir_name + "pixel_pos.txt")
    mesh = meshio.read(dir_name + "surface_fixed.obj")
    wave_number = float(np.loadtxt(dir_name + "omega.txt")) / 343
    surf_neumann = np.loadtxt(dir_name + "surf_neumann.txt")
    dim = int((len(pixel_pos) / 6)**0.5)
    pixel_pos = pixel_pos.reshape((6 * dim * dim, 3))
    vertices = mesh.points
    faces = mesh.cells[0].data

    bem_model = BEMModel(vertices, faces, wave_number)
    bem_model.boundary_equation_solve(surf_neumann)
    ffat = bem_model.potential_solve(pixel_pos)
    ffat = norm_ffat(abs(ffat), dim)
    pppm_ffat = norm_ffat(np.loadtxt(dir_name + "pppm_ffat.txt"), dim)
    ghost_ffat1 = norm_ffat(np.loadtxt(dir_name + "ghost1_ffat.txt"), dim)
    ghost_ffat2 = norm_ffat(np.loadtxt(dir_name + "ghost2_ffat.txt"), dim)
    concated = np.concatenate(
        (ffat, pppm_ffat, ghost_ffat1, ghost_ffat2), axis=1)
    fig = plt.figure(figsize=(10, 5))
    # hide axes
    fig.patch.set_visible(False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(concated, cmap='jet', alpha=0.7, interpolation='bilinear')
    plt.savefig(img_dir + "/" + names[-3] + "_" + names[-2] +
                "ffat.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    return SNR(ffat, pppm_ffat), SNR(ffat, ghost_ffat1), SNR(ffat, ghost_ffat2)


root_dir = "../../dataset/acoustic/" + obj_name + "/output/"
img_dir = './img/' + obj_name
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
time_data = {"pppm": [], "ghost1": [], "ghost2": []}
SNR_data = {"pppm": [], "ghost1": [], "ghost2": []}
sub_dir_list = glob(root_dir + "*/")
res_list = [int(sub_dir.split("/")[-2]) for sub_dir in sub_dir_list]
res_list.sort()
from tqdm import tqdm
for i in tqdm(res_list):
    pppm_time = 0
    ghost1_time = 0
    ghost2_time = 0
    SNR_pppm = 0
    SNR_ghost1 = 0
    SNR_ghost2 = 0
    mode_idx_dir_list = glob(root_dir + str(i) + "/*/")
    mode_idx_list = [int(mode_idx.split("/")[-2])
                     for mode_idx in mode_idx_dir_list]
    for j in mode_idx_list:
        dir_name = root_dir + str(i) + "/" + str(j) + "/"
        SNR_list = process_acoustic_data(dir_name)
        pppm_time += np.loadtxt(dir_name + "pppm_time.txt")
        ghost1_time += np.loadtxt(dir_name + "ghost1_time.txt")
        ghost2_time += np.loadtxt(dir_name + "ghost2_time.txt")
        SNR_pppm += SNR_list[0]
        SNR_ghost1 += SNR_list[1]
        SNR_ghost2 += SNR_list[2]
    SNR_data["pppm"].append(SNR_pppm / len(mode_idx_list))
    SNR_data["ghost1"].append(SNR_ghost1 / len(mode_idx_list))
    SNR_data["ghost2"].append(SNR_ghost2 / len(mode_idx_list))
    time_data["pppm"].append(pppm_time / len(mode_idx_list))
    time_data["ghost1"].append(ghost1_time / len(mode_idx_list))
    time_data["ghost2"].append(ghost2_time / len(mode_idx_list))

plt.figure(figsize=(15, 8))
plt.plot(res_list, time_data["pppm"], label="PPPM")
plt.plot(res_list, time_data["ghost1"], label="Ghost1")
plt.plot(res_list, time_data["ghost2"], label="Ghost2")
plt.legend()
plt.ylabel("Time (s)")
plt.xlabel("Resolution")
plt.savefig(img_dir + "/acoustic_time.png", dpi=300,
            bbox_inches='tight', pad_inches=0)
plt.close()

plt.figure(figsize=(15, 8))
plt.plot(res_list, SNR_data["pppm"], label="PPPM")
plt.plot(res_list, SNR_data["ghost1"], label="Ghost Cell 1st")
plt.plot(res_list, SNR_data["ghost2"], label="Ghost Cell 2nd")
plt.legend()
plt.ylabel("SNR")
plt.xlabel("Resolution")
plt.savefig(img_dir + "/acoustic_SNR.png", dpi=300,
            bbox_inches='tight', pad_inches=0)
plt.close()
