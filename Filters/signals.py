import functions as func
from functions import np, plt, sig, rnd


# Параметры сигнала
ampl = 5    # Амплитуда
phi = 0     # Сдвиг по фазе
f = 0.1     # Частота, Гц
fd = 4      # Частота дискретизации, Гц
n = 1000    # Количеств отсчётов

noiseCount = 10

# Генерация сигнала
t = np.arange(0, n)
real = ampl * np.sin(2 * np.pi * f * t / fd + phi)
real = func.addSomeNoise(real, noiseCount, t)
imag = ampl * np.sin(2 * np.pi * f * t / fd + phi + np.pi / 2)
imag = func.addSomeNoise(imag, noiseCount, t)

plt.plot(real, 'g')
plt.plot(imag, 'r')
plt.title('Исходный сигнал')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# ДПФ
dft_ = func.dft(real + imag * 1.0j)
plt.plot(np.abs(dft_))
plt.title('Модуль ДПФ')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

N = 100
fc = 0.1  # Частота полосы затухания, Гц
Fs = 1  # Частота полосы пропускания, Гц
w_c = 2 * fc / Fs

ff = 1 / (2 * np.pi / f)  # Частота среза
# plt.plot(func.dftForward(func.fir(dft_, n, N, ff, Fs, fc)))
# plt.show()

# tabs = func.compute_fir_filter_coefficients(N, w_c, sr)
# tabs = func.calculateFirCoeffs(w_c, N)
tabs = sig.firwin(N, w_c)

res = func.firFilter(real + imag * 1.0j, tabs)
# res = func.dftForward(res)
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
