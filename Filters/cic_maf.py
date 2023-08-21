import numpy as np
import functions as func
import matplotlib.pyplot as plt


def maf_conv(signal, m=2):
    """
    Calculate moving average filter via convolution
    Parameters
    ----------
    m : int
        moving average step
    """
    coe = np.ones(m) / m
    return np.convolve(signal, coe, mode='same')    


def maf(signal, m=2):
    """
    Calculate moving average filter via convolution
    Parameters
    ----------
    m : int
        moving average step
    """
    # coe = np.ones(m) / m
    coe = func.compute_fir_filter_coefficients(m, 10.0, sample_rate)
    return func.convolution(signal, coe)


sample_rate = 100.0
nsamples = 200
ampl = 15
freq = 5
noiseCount = 100

t = np.arange(nsamples) / sample_rate
t[0] = sample_rate

signal = ampl * np.sin(2 * np.pi * freq * t)
signal = func.addSomeNoise(signal, noiseCount, t, freq, ampl)

m = 10

filtered = maf_conv(signal, m)

plt.figure()
plt.plot(signal, 'g')
plt.plot(filtered, 'b', linewidth=3)
plt.title(f'maf_conv, m = {m}')

filtered = maf(signal, m)

plt.figure()
plt.plot(signal, 'g')
plt.plot(filtered, 'b', linewidth=3)
plt.title(f'maf (FIR), m = {m}')

hfq = np.zeros(signal.size)
for i in range(signal.size):
        if i == 0:
            hfq[i] = 1
        else:
            hfq[i] = np.abs(np.sin(np.pi * m * i / 2 / signal.size) / m /
                               np.sin(np.pi * i / 2 / signal.size))

plt.figure()
plt.plot(hfq)
plt.title('freq response')

plt.show()