import numpy as np
import random as rnd
import window_functions as wifu


# Добавление шума
def addSomeNoise(f, n, t, freq, ampl):
    for i in range(n):
        f += rnd.random() * np.cos(2 * np.pi * rnd.random() * freq * t * i)
    return f


def getBandPassFilterCoeffs(filterSize, lowCutoffFreq, highCutoffFreq, sampleRate):
    coefficients = np.zeros((filterSize))

    center = filterSize / 2

    lowNormalizedCutoffFreq = lowCutoffFreq / sampleRate
    highNormalizedCutoffFreq = highCutoffFreq / sampleRate

    for n in range(filterSize):
        offset = n - center

        if offset == 0:
            coeff = 2 * (highNormalizedCutoffFreq - lowNormalizedCutoffFreq)
        else:
            coeff = (np.sin(2 * np.pi * highNormalizedCutoffFreq * offset) - np.sin(2 * np.pi * lowNormalizedCutoffFreq * offset)) / (np.pi * offset)
        
        coeff *= wifu.flapTopWindow(n, filterSize)

        coefficients[n] = coeff

    return coefficients


def applyFirFilter(signal, coeffs):
    N = signal.size
    M = coeffs.size
    newSize = (N + M) if (M % 2 == 0) else (N + M - 1)
    y = np.zeros((newSize))
    for n in range(newSize):
        for k in range(M):
            if n - k >= 0 and n - k < N:
                y[n] += coeffs[k] * signal[n - k]

    return y


def compensatePhaseDelay(signal, filterSize):
    withoutDelays = signal
    withoutDelays = withoutDelays[filterSize//2:signal.size - filterSize//2]
    return withoutDelays


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
def dftInverse(x):
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


# Свёртка
def convolution(signal, kernel):
    signal_length = len(signal)
    kernel_length = len(kernel)
    result_length = signal_length + kernel_length - 1
    result = [0] * result_length
    
    # Определение центра свертки
    center = kernel_length // 2
    
    # Перебор всех элементов результата
    for i in range(result_length):
        # Перебор всех элементов сигнала и ядра для текущего элемента результата
        for j in range(signal_length):
            # Проверка, чтобы индекс находился в пределах массива сигнала
            if i - j >= 0 and i - j < kernel_length:
                # Выполнение свертки путем умножения элементов сигнала и ядра
                result[i] += signal[j] * kernel[i - j]
    
    return result
