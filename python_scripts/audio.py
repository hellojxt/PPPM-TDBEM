import IPython
from numba import njit
import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io.wavfile import write


def show_audio(x, sr):
    x = x.astype('float32')
    x /= x.max()
    return IPython.display.Audio(x, rate=sr)


def save_audio(x, sr, path):
    x = x.astype('float32')
    x /= x.max()
    # resample to 44100
    if sr != 44100:
        x = scipy.signal.resample(x, int(len(x) * 44100 / sr))
        sr = 44100
    write(path, sr, x)


@njit()
def _IIR(f, e, theta, r, wwd):
    signal = np.zeros_like(f)
    for idx in range(len(signal)):
        if idx < 3:
            continue
        signal[idx] = 2*e*np.cos(theta)*signal[idx - 1] - \
            e**2*signal[idx - 2] + \
            2*f[idx-1]*(e*np.cos(theta+r)-e**2*np.cos(2*theta + r))/(3*wwd)
    return signal


def IIR(f, val, alpha, beta, h):  # h是时间步长,f是冲激大小(似乎？)
    d = 0.5*(alpha + beta*val)  # 一次项damping
    e = np.exp(-d*h)
    wd = (val - d**2)**0.5  # 频率f=wd/2pi
    theta = wd*h
    w = (val**0.5)
    r = np.arcsin(d / w)
    return _IIR(f, e, theta, r, w*wd)
