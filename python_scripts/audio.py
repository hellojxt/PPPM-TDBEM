import IPython


def show_audio(x, sr):
    x = x.astype('float32')
    x /= x.max()
    return IPython.display.Audio(x, rate=sr)
