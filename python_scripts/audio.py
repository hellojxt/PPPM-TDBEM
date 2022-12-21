import IPython
from numba import njit
import numpy as np


def show_audio(x, sr):
    x = x.astype('float32')
    x /= x.max()
    return IPython.display.Audio(x, rate=sr)


@njit()
def _IIR(f, e, theta, r, wwd):
    print("coeff1:", 2*e*np.cos(theta), "coeff2:", -e**2, "coeff3:",
          2*(e*np.cos(theta+r)-e**2*np.cos(2*theta + r))/(3*wwd))
    signal = np.zeros_like(f)
    for idx in range(len(signal)):
        if idx < 3:
            continue
        signal[idx] = 2*e*np.cos(theta)*signal[idx - 1] - \
            e**2*signal[idx - 2] + \
            2*f[idx-1]*(e*np.cos(theta+r)-e**2*np.cos(2*theta + r))/(3*wwd)
        print(signal[idx], "=", 2*e*np.cos(theta), "*", signal[idx - 1], "-", e**2, "*",
              signal[idx - 2], "+", 2*f[idx-1], "*", (e*np.cos(theta+r)-e**2*np.cos(2*theta + r))/(3*wwd))
    return signal


def IIR(f, val, alpha, beta, h):  # h是时间步长,f是冲激大小(似乎？)
    d = 0.5*(alpha + beta*val)  # 一次项damping
    e = np.exp(-d*h)
    wd = (val - d**2)**0.5  # 频率f=wd/2pi
    theta = wd*h
    w = (val**0.5)
    r = np.arcsin(d / w)
    return _IIR(f, e, theta, r, w*wd)
