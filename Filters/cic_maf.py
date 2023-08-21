import numpy as np
import functions as func
import matplotlib.pyplot as plt
from cic_functions import maf, maf_conv


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

filtered = maf(signal, sample_rate, m)

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