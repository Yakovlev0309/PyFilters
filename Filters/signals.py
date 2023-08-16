import functions as func
from functions import np, plt, sig, rnd
import scipy


def calculate_cutoff_frequency(signal, sampling_freq):
    spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(spectrum), 1/sampling_freq)
    magnitudes = np.abs(spectrum)
    max_magnitude = np.max(magnitudes)
    cutoff_freq_index = np.where(magnitudes == max_magnitude)[0]
    cutoff_freq = frequencies[cutoff_freq_index]
    return cutoff_freq[0]

def calculate_passband_frequency(cutoff_freq):
    passband_freq = cutoff_freq * 0.8  # Пример: 80% частоты среза
    return passband_freq

def calculate_stopband_frequency(cutoff_freq):
    stopband_freq = cutoff_freq * 1.2  # Пример: 120% частоты среза
    return stopband_freq

# Параметры сигнала
ampl = 25    # Амплитуда
freq = 1     # Частота, Гц
sampleRate = 100    # Частота дискретизации, Гц
timeDuration = 10    # Количество временных отсчётов

noiseCount = 3

# Генерация сигнала
t = np.arange(0, timeDuration * sampleRate)
real = ampl * np.sin(2 * np.pi * freq * t / sampleRate)
imag = ampl * np.cos(2 * np.pi * freq * t / sampleRate)
real = func.addSomeNoise(real, noiseCount, t, freq, ampl)
imag = func.addSomeNoise(imag, noiseCount, t, freq, ampl)
signal = real + imag * 1.0j;

# ДПФ
# ft = func.dft(signal)
ft = np.fft.fft(signal)

filterSize = 100
# fc = 0.1  # Частота полосы затухания, Гц
# Fs = 1  # Частота полосы пропускания, Гц
# cutoffFreq = 2 * fc / Fs

cutoffFreq = 1 / (2 * np.pi / freq)  # Частота среза
# plt.plot(func.dftForward(func.fir(dft_, n, N, ff, Fs, fc)))
# plt.show()

# cutoffFreq = calculate_cutoff_frequency(signal, sampleRate)
# passbandFreq = calculate_passband_frequency(cutoffFreq)
# stopbandFreq = calculate_stopband_frequency(cutoffFreq)

# cutoffFreq = sampleRate * 0.4
passbandFreq = cutoffFreq
stopbandFreq = passbandFreq / 2.0
# attenuationFreq = (stopbandFreq - passbandFreq) / 2.0

# coeffs = func.compute_fir_filter_coefficients(filterSize, cutoffFreq, sampleRate)
# coeffs = func.calculateFirCoeffs(cutoffFreq, filterSize)
coeffs = sig.firwin(filterSize, cutoffFreq, fs=sampleRate)
# coeffs = func.getFilterCoeffs(filterSize, sampleRate, passbandFreq, stopbandFreq)

# fFiltered = func.firFilter(ft, coeffs)
fFiltered = func.fir(ft, coeffs)

# filtered = func.dftForward(fFiltered)
filtered = np.fft.ifft(fFiltered)


plt.subplot(2, 1, 1)
plt.plot(np.abs(coeffs))
plt.title('Коэффициенты фильтра')
plt.grid()
plt.subplot(2, 1, 2)
# plt.plot(np.abs(func.dft(coeffs)))
plt.plot(np.abs(np.fft.fft(coeffs)))
plt.title('АЧХ фильтра')
plt.grid()
plt.show()

plt.subplot(2, 1, 1)
plt.plot(real, 'g')
plt.plot(imag, 'r')
plt.title('Исходный сигнал')
plt.xlabel('Время')
plt.ylabel('Амплитуда')
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(real, 'g')
plt.plot(imag, 'r')
plt.plot(filtered, 'b')
plt.title('Отфильтрованный сигнал')
plt.xlabel('Время')
plt.ylabel('Амплитуда')
plt.grid()
plt.show()

plt.subplot(2, 1, 1)
plt.plot(np.abs(ft))
plt.title('Частотная область')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(np.abs(fFiltered))
plt.title('Отфильтрованный сигнал')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.grid()
plt.show()

plt.subplot(2, 1, 1)
plt.plot(20 * np.log10(np.abs(signal)))
plt.title('До')
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(20 * np.log10(np.abs(filtered)))
plt.title('После')
plt.grid()
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
