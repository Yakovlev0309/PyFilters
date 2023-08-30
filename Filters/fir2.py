
# Финальная версия КИХ-фильтра

import numpy as np
import functions as func
import matplotlib.pyplot as plt


sample_rate = 100.0
nsamples = 200
ampl = 15
freq = 5
noiseCount = 100
freq_factor = 5

cutoff_freq = 10.0

filter_size = 100

t = np.arange(nsamples) / sample_rate
t[0] = sample_rate

signal = ampl * np.sin(2 * np.pi * freq * t)
signal = func.addSomeNoise(signal, noiseCount, t, freq, freq_factor)

# coeffs = func.getBandPassFilterCoeffs(filter_size, 0, cutoff_freq, sample_rate)
coeffs = func.getBandPassFilterCoeffs(filter_size, cutoff_freq, sample_rate / 2.0, sample_rate)
filtered = func.applyFirFilter(signal, coeffs)
filtered = func.compensatePhaseDelay(filtered, filter_size)

fSignal = func.dft(signal)
fFiltered = func.dft(filtered)

plt.figure()
plt.plot(signal, 'g')
plt.plot(filtered, 'b', linewidth=2)
plt.grid()

plt.figure()
plt.plot(fSignal)
plt.title('signal freq response')
plt.grid()

plt.figure()
plt.plot(fFiltered)
plt.title('filtered freq response')
plt.grid()
plt.show()