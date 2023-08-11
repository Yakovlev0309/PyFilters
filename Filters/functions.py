import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import scipy.signal as sig


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


# КИХ-фильтр
def fir(In, sizeIn):
    N = 20;  
    Fd = 40;  # Частота дискретизации
    Fs = 1000;  # Частота полосы пропускаяния
    Fx = 100;  # Частота полосы затухания

    H = np.full(N, 0.0);  # Импульсная характеристика фильтра
    H_id = np.full(N, 0.0);  # Идеальная импульсная характеристика фильтра
    W = np.full(N, 0.0);  # Весовая функция

    Fc = (Fs + Fx) / (2 * Fd);

    for i in range(N):
        if (i == 0):
            H_id[i] = 2 * np.pi * Fc;
        else:
            H_id[i] = np.sin(2 * np.pi * Fc * i) / (np.pi * i);
        # Весовая функция Блэкмана
        W[i] = 0.42 - 0.5 * np.cos((2 * np.pi * i) /(N - 1)) + 0.08 * np.cos((4 * np.pi * i) / (N - 1));
        H[i] = H_id[i] * W[i];

    # Нормировка импульсной характеристики
    SUM = 0.0
    for i in range(N):
        SUM += H[i]
    for i in range(N):
        H[i] /= SUM  # Сумма коэффициентов равна 1
    
    # Фильтрация
    Out = np.arange(sizeIn)
    for i in range(sizeIn):
        Out[i] = 0
        for j in range(N - 1):
            if (i - j >= 0):
                Out[i] += H[j] * In[i - j]

    return Out