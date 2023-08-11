from scipy.signal import kaiserord, lfilter, firwin
import numpy as np

def filter(signal):
    sample_rate = 100.0
    
    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2.0

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0/nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    cutoff_hz = 10.0

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    window = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

    print(window)

    # Use lfilter to filter x with the FIR filter.
    print(signal)
    filtered_x = lfilter(window, 1.0, signal)
    return filtered_x


def filter2(In, Out, sizeIn):
    N = 20;  
    Fd = 2000;  # Частота дискретизации
    Fs = 20;  # Частота полосы пропускаяния
    Fx = 50;  # Частота полосы затухания

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
    for i in range(sizeIn):
        Out[i] = 0
        for j in range(N - 1):
            if (i - j >= 0):
                Out[i] += H[j] * In[i - j]

    return Out
