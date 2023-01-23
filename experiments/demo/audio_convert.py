import sys
sys.path.append('../../python_scripts')
from visulize import viewer
from audio import show_audio, IIR, save_audio
import numpy as np
import meshio
import matplotlib.pyplot as plt
import os
# check the mesh and modes
output_dir = sys.argv[1]
y = np.loadtxt(output_dir + '/result.txt')
sr = int(y[0])
y = y[1:]
y /= abs(y).max()
save_audio(y, sr, output_dir + '/result.wav')
