import numpy as np
import functions as func
import matplotlib.pyplot as plt
from cic_functions import CicFilter, maf, maf_conv


def moving_average_filter(input, window_size):
    output = []
    for i in range(len(input)):
        if i < window_size - 1:
            output.append(sum(input[:i+1]) / (i+1))
        else:
            output.append(sum(input[i-window_size+1:i+1]) / window_size)
    return output


sample_rate = 100.0
nsamples = 200
ampl = 15
freq = 5
noiseCount = 100

cutoffFreq = 10.0

t = np.arange(nsamples) / sample_rate
t[0] = sample_rate

signal = ampl * np.sin(2 * np.pi * freq * t)
signal = func.addSomeNoise(signal, noiseCount, t, freq, ampl)

filterSize = 100

decimation = 2
filterOrder = 2

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(signal)
plt.title('(1) signal')
plt.grid()

cicFilter = CicFilter(signal)
filtered = cicFilter.decimator(decimation, filterOrder)
plt.subplot(3, 1, 2)
plt.plot(filtered)
plt.title(f'(2) decimator = {decimation}')
plt.grid()

coeffs = func.getBandPassFilterCoeffs(filterSize, 0, cutoffFreq, sample_rate)
filtered = func.applyFirFilter(filtered, coeffs)
filtered = func.compensatePhaseDelay(filtered, filterSize)
plt.subplot(3, 1, 3)
plt.plot(filtered)
plt.title('(3) filtered')
plt.grid()

plt.figure()
plt.plot(signal, 'g')
plt.plot(np.linspace(0, signal.size, filtered.size), filtered, 'b', linewidth=2)
plt.grid()
plt.show()