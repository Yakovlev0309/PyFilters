import numpy as np
import pylab as plt
from scipy.fftpack import rfft, irfft, fftfreq

import scipy as sp

time   = np.linspace(0,10,2000) # Время
signal = np.cos(5*np.pi*time) # Сигнал (синусоида)

w = fftfreq(signal.size, d=time[1]-time[0])
f_signal = rfft(signal)

# If our original signal time was in seconds, this is now in Hz    
cut_f_signal = f_signal.copy()
cut_f_signal[(w<6)] = 0

cut_signal = irfft(cut_f_signal)

plt.subplot(211)
plt.title("signal")
plt.plot(time, signal)
plt.subplot(212)
plt.title("rfft")
plt.plot(w, f_signal)

# plt.xlim(0,10)
# plt.subplot(223)
# plt.plot(w,cut_f_signal)
# plt.xlim(0,10)
# plt.subplot(224)
# plt.plot(time,cut_signal)
plt.show()
