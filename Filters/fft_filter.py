import numpy as np
import functions as func
import matplotlib.pyplot as plt


def butterworth_lowpass_filter(signal, cutoff_freq, sampling_freq):
    # Применение фильтра Баттерворта низких частот к сигналу
    # signal - исходный сигнал
    # cutoff_freq - частота среза фильтра (в герцах)
    # sampling_freq - частота семплирования сигнала (в герцах)

    # Выполнение быстрого преобразования Фурье (FFT) для сигнала
    fft_signal = func.dft(signal)

    # Вычисление частот
    freq = np.fft.fftfreq(len(signal), 1 / sampling_freq)

    # Создание фильтра Баттерворта низких частот
    filter = np.zeros(len(signal))
    filter[np.abs(freq) <= cutoff_freq] = 1

    # Умножение FFT сигнала на фильтр низких частот
    filtered_signal = func.dftInverse(fft_signal * filter)

    # Возвращение отфильтрованного сигнала
    return np.real(filtered_signal)


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

filtered = butterworth_lowpass_filter(signal, cutoffFreq, sample_rate)


plt.plot(signal, 'g')
plt.plot(filtered, 'b', linewidth=3)
plt.show()