import IPython


def show_audio(x, sr):
    return IPython.display.Audio(x, rate=sr)
