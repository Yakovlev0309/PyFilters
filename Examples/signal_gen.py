import numpy as np

def getSinSignal(begin, end, h, k, b):
    signal = np.empty(end - begin)
    step = begin
    while step <= end:
        np.append(signal, {step, k * np.sin(step) + b})
        step += h
    return signal

def getComplexSin(ampl, begin, end, rate):
    x = np.arange(begin, end, rate)
    # real = ampl * np.sin(x)
    # imag = ampl * np.sin(x + np.pi / 2)

    real = np.cos(2*np.pi*0.5) + 0.2*np.sin(2*np.pi*2.5+0.1) + \
        0.2*np.sin(2*np.pi*15.3) + 0.1*np.sin(2*np.pi*16.7+ 0.1) + \
            0.1*np.sin(2*np.pi*23.45+.8 * x)
    imag = np.cos(2*np.pi*0.5) + 0.2*np.sin(2*np.pi*2.5+0.1) + \
        0.2*np.sin(2*np.pi*15.3) + 0.1*np.sin(2*np.pi*16.7 + 0.1) + \
            0.1*np.sin(2*np.pi*23.45+.8 * x)
    return [x, real, imag]