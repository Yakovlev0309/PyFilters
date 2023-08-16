import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import scipy.signal as sig


# Добавление шума
def addSomeNoise(f, n, t, freq, ampl):
    for i in range(n):
        f += rnd.random() * np.cos(2 * np.pi * rnd.random() * freq * t * i)
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


# ДПФ обратное
def dftForward(x):
    size = len(x)
    result = [complex(0.0, 0.0) for _ in range(size)]

    for k in range(size):
        for n in range(size):
            angle = 2 * np.pi * k * n / size
            cosine = np.cos(angle)
            sine = np.sin(angle)

            result[k] += x[n] * complex(cosine, -sine)

        # Нормализация
        result[k] /= size
    return result


# Расчёт коэффициентов КИХ-фильтра
def calculateFirCoeffs(cutoff_freq, num_taps):
    coefficients = []
    N = num_taps - 1
    for n in range(num_taps):
        if n == N/2:
            coefficients.append(2 * cutoff_freq)
        else:
            coefficient = (2 * cutoff_freq * (n - N/2)) / (n - N/2)
            coefficients.append(coefficient)
    return coefficients


def compute_fir_filter_coefficients(num_taps, cutoff_freq, sampling_rate):
    coefficients = []

    # Вычисляем центральный индекс
    center = num_taps // 2

    # Вычисляем значение сдвига частоты среза
    normalized_freq = cutoff_freq / sampling_rate

    # Вычисляем коэффициенты фильтра
    for n in range(num_taps):
        # Вычисляем смещение относительно центра
        offset = n - center

        if offset == 0:
            coefficient = 2 * normalized_freq
        else:
            coefficient = np.sin(2 * np.pi * normalized_freq * offset) / (np.pi * offset)
        
        # Добавляем коэффициент в массив
        coefficients.append(coefficient)
    
    return coefficients


# КИХ-фильтр
def firFilter(signal, coefficients):
    output = []
    for i in range(len(signal)):
        value = 0
        for j in range(len(coefficients)):
            if i - j >= 0:
                value += signal[i - j] * coefficients[j]
        output.append(value)
    return output


def getFilterCoeffs(filterSize, sampleRate, passbandFreq, stopbandFreq):
    H = np.full(filterSize, 0.0)  # Импульсная характеристика фильтра
    H_id = np.full(filterSize, 0.0)  # Идеальная импульсная характеристика фильтра
    W = np.full(filterSize, 0.0)  # Весовая функция

    Fc = (passbandFreq + stopbandFreq) / (2 * sampleRate)  # Расчёт импульсной характеристики фильтра

    for i in range(filterSize):
        # Low-pass
        if (i == 0):
            H_id[i] = 2 * np.pi * Fc
        else:
            H_id[i] = np.sin(2 * np.pi * Fc * i) / (np.pi * i)

        # Весовая функция Блэкмана
        W[i] = 0.42 + 0.5 * np.cos((2 * np.pi * i) / (filterSize - 1)) + 0.08 * np.cos((4 * np.pi * i) / (filterSize - 1))
        H[i] = H_id[i] * W[i]

    # Нормировка импульсной характеристики
    SUM = 0.0
    for i in range(filterSize):
        SUM += H[i]
    for i in range(filterSize):
        H[i] /= SUM  # Сумма коэффициентов равна 1
    
    return H


# КИХ-фильтр
def fir(In, coeffs):
    filterSize = len(coeffs)
    sizeIn = len(In)
    # Фильтрация
    Out = np.arange(0, sizeIn, dtype=complex)
    for i in range(sizeIn):
        Out[i] = 0
        for j in range(filterSize - 1):
            if (i - j >= 0):
                Out[i] += coeffs[j] * In[i - j]

    return np.asarray(Out, complex)
