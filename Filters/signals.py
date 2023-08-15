import functions as func
from functions import np, plt, sig, rnd


# Параметры сигнала
ampl = 30    # Амплитуда
freq = 1000     # Частота, Гц
sampleRate = 40000      # Частота дискретизации, Гц
timeDuration = 1000    # Количество временных отсчётов

noiseCount = 100

# Генерация сигнала
t = np.arange(0, timeDuration)
real = ampl * np.sin(2 * np.pi * freq * t / sampleRate)
real = func.addSomeNoise(real, noiseCount, t, freq, ampl)
imag = ampl * np.sin(2 * np.pi * freq * t / sampleRate + np.pi / 2)
imag = func.addSomeNoise(imag, noiseCount, t, freq, ampl)
signal = real + imag * 1.0j;

plt.plot(real, 'g')
plt.plot(imag, 'r')
plt.title('Исходный сигнал')
plt.xlabel('Время')
plt.ylabel('Амплитуда')
plt.grid()
plt.show()

# ДПФ
dft_ = func.dft(signal)
plt.plot(np.abs(dft_))
plt.title('Модуль ДПФ')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.show()

filterSize = 100
# fc = 0.1  # Частота полосы затухания, Гц
# Fs = 1  # Частота полосы пропускания, Гц
# cutoffFreq = 2 * fc / Fs

# ff = 1 / (2 * np.pi / freq)  # Частота среза
# plt.plot(func.dftForward(func.fir(dft_, n, N, ff, Fs, fc)))
# plt.show()

cutoffFreq = sampleRate / 2.0 * 0.7
passbandFreq = cutoffFreq
stopbandFreq = sampleRate / 2.0
attenuationFreq = (stopbandFreq - passbandFreq) / 2.0

# tabs = func.compute_fir_filter_coefficients(filterSize, cutoffFreq, sampleRate)
# tabs = func.calculateFirCoeffs(cutoffFreq, filterSize)
# tabs = sig.firwin(filterSize, cutoffFreq)

# fFiltered = func.firFilter(dft_, tabs)
fFiltered = func.fir(dft_, len(dft_), filterSize, sampleRate, passbandFreq, attenuationFreq)
plt.plot(np.abs(fFiltered))
plt.title('Отфильтрованный сигнал')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.show()

filtered = func.dftForward(fFiltered)
plt.plot(filtered)
plt.title('Отфильтрованный сигнал')
plt.xlabel('Время')
plt.ylabel('Амплитуда')
plt.show()



# w, h = sig.freqz(tabs, worN = 2000)
# w = Fs * w / (2 * np.pi)  # Frequency (Hz)
# h_db = 20 * np.log10(abs(h))  # Magnitude (dB)

# plt.plot(w, h_db)
# plt.title('FIR filter response')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude (dB)')
# plt.show()

# plt.plot(h)
# plt.title('Переходная характеристика')
# plt.show()
