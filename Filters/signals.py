import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import scipy.signal as sig

# Добавление шума
def addSomeNoise(f, n):
    for i in range(n):
        f += rnd.random() * np.cos(2 * np.pi * rnd.random() * t)
    return f


# ДПФ
def dft(signal):
    N = len(signal)
    spectrum = []
    for k in range(N):
        value = 0
        for n in range(N):
            angle = 2 * np.pi * k * n / N
            value += signal[n] * np.exp(-1j * angle)
        spectrum.append(value)   
    return spectrum


# Параметры сигнала
ampl = 12
phi = 0
f = 0.01  # Гц
n = 100

noiseCount = 10

# Генерация сигнала
t = np.arange(0, n)
real = ampl * np.sin(2 * np.pi * f * t + phi)
real = addSomeNoise(real, noiseCount)
imag = ampl * np.sin(2 * np.pi * f * t + phi + np.pi / 2)
imag = addSomeNoise(imag, noiseCount)

plt.plot(real, 'g')
plt.plot(imag, 'r')
plt.title('Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

dft_ = dft(real)
plt.plot(dft_)
plt.title('DFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

N = 20
fc = 100
Fs = 1000
w_c = 2 * fc / Fs
t = sig.firwin(N, f)
w, h = sig.freqz(t, worN = n)
w = Fs * w / (2 * np.pi)  # Frequency (Hz)'
h_db = 20 * np.log10(abs(h))  # Magnitude (dB)

plt.plot(w, h_db)
plt.title('FIR filter response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.show()

plt.plot(h)
plt.title('h')
plt.show()
