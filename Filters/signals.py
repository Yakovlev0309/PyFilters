import functions as func
from functions import np, plt, sig, rnd


# Добавление шума
def addSomeNoise(f, n):
    for i in range(n):
        f += rnd.random() * np.cos(2 * np.pi * rnd.random() * t)
    return f


# Параметры сигнала
ampl = 5
phi = 0
f = 0.1  # Гц
sr = 4  # Гц
n = 1000

noiseCount = 10

# Генерация сигнала
t = np.arange(0, n)
real = ampl * np.sin(2 * np.pi * f * t / sr + phi)
real = addSomeNoise(real, noiseCount)
imag = ampl * np.sin(2 * np.pi * f * t / sr + phi + np.pi / 2)
imag = addSomeNoise(imag, noiseCount)

plt.plot(real, 'g')
plt.plot(imag, 'r')
plt.title('Исходный сигнал')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# ДПФ
dft_ = func.dft(real + imag * 1.0j)
# plt.plot(dft_)
# plt.title('DFT')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.show()

plt.plot(np.abs(dft_))
plt.title('Модуль ДПФ')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

plt.plot(func.dftForward(dft_))
plt.title('Обратное ДПФ')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# ff = 1 / (2 * np.pi / f)
# coeffs = func.calculateFirCoeffs(ff, 50)
# plt.plot(coeffs)
# plt.title('FIR filter coeffs')
# plt.show()

N = 32
fc = 0.1  # Частота полосы затухания
Fs = 1  # Частота полосы пропускания
w_c = 2 * fc / Fs
tabs = sig.firwin(N, w_c)

res = func.firFilter(real + imag * 1.0j, tabs)
plt.plot(real, 'g')
plt.plot(res, 'b')
plt.title('Исходный + отфильтрованный сигналы')
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
