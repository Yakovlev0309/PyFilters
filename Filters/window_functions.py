import numpy as np


def blackmanWindow(n, windowSize):
    # При alpha = 0.16
    a0 = 0.42
    a1 = 0.5
    a2 = 0.08
    angle = np.pi * n / (windowSize - 1)
    return a0 - a1 * np.cosl(2 * angle) + a2 * np.cosl(4 * angle)


def blackmanHarrisWindow( n, windowSize):
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    angle = np.pi * n / (windowSize - 1)
    return a0 - a1 * np.cos(2 * angle) + a2 * np.cos(4 * angle) - a3 * np.cos(6 * angle)


def hammingWindow(n, windowSize):
    a0 = 0.54
    a1 = 0.46
    angle = np.pi * n / (windowSize - 1)
    return a0 - a1 * np.cos(2 * angle)


def nuttallWindow(n, windowSize):
    a0 = 0.355768
    a1 = 0.487396
    a2 = 0.144232
    a3 = 0.012604
    angle = np.pi * n / (windowSize - 1)
    return a0 - a1 * np.cos(2 * angle) + a2 * np.cos(4 * angle) - a3 * np.cos(6 * angle)


def blackmanNuttallWindow(n, windowSize):
    a0 = 0.3635819
    a1 = 0.4891775
    a2 = 0.1365995
    a3 = 0.0106411
    angle = np.pi * n / (windowSize - 1)
    return a0 - a1 * np.cos(2 * angle) + a2 * np.cos(4 * angle) - a3 * np.cos(6 * angle)


def flapTopWindow(n, windowSize):
    a0 = 0.21557895
    a1 = 0.41663158
    a2 = 0.277263158
    a3 = 0.083578947
    a4 = 0.006947368
    angle = np.pi * n / (windowSize - 1)
    return a0 - a1 * np.cos(2 * angle) + a2 * np.cos(4 * angle) - a3 * np.cos(6 * angle) + a4 * np.cos(8 * angle)