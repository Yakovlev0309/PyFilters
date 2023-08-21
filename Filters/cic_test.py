import numpy as np
import functions as func
import matplotlib.pyplot as plt
from cic_functions import CicFilter, maf, maf_conv


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

filterSize = 5

decimation = 2
interpolation = 2
filterOrder = 1


plt.figure()
plt.subplot(2, 2, 1)
plt.plot(signal)
plt.title('(1) signal')
plt.grid()

cicFilter = CicFilter(signal)
filtered = cicFilter.interpolator(interpolation, filterOrder)
cicFilter.x = filtered
plt.subplot(2, 2, 2)
plt.plot(filtered)
plt.title(f'(2) decimation = {decimation}')
plt.grid()

filtered = maf(filtered, sample_rate, filterSize)
cicFilter.x = filtered
plt.subplot(2, 2, 3)
plt.plot(filtered)
plt.title(f'(3) filter size = {filterSize}')
plt.grid()

filtered = cicFilter.decimator(decimation, filterOrder)
cicFilter.x = filtered
plt.subplot(2, 2, 4)
plt.plot(filtered)
plt.title(f'(4) interpolation = {interpolation}')
plt.grid()


plt.figure()
plt.plot(signal, 'g')
plt.plot(filtered, 'b', linewidth=2)
plt.grid()
plt.show()