import numpy as np
import functions as func
import matplotlib.pyplot as plt

class CicFilter:
    """
    Cascaded Integrator-Comb (CIC) filter is an optimized class of
    finite impulse response (FIR) filter.
    CIC filter combines an interpolator or decimator, so it has some
    parameters:

    R - decimation or interpolation ratio,
    N - number of stages in filter (or filter order)
    M - number of samples per stage (1 or 2)*

    * for this realisation of CIC filter just leave M = 1.

    CIC filter is used in multi-rate processing. In hardware
    applications CIC filter doesn't need multipliers, just only
    adders / subtractors and delay lines.

    Equation for 1st order CIC filter:
    y[n] = x[n] - x[n-RM] + y[n-1].


    Parameters
    ----------
    x : np.array
        input signal
    """

    def __init__(self, x):
        self.x = x

    def decimator(self, r, n):
        """
        CIC decimator: Integrator + Decimator + Comb

        Parameters
        ----------
        r : int
            decimation rate
        n : int
            filter order
        """

        # integrator
        y = self.x[:]
        for i in range(n):
            y = np.cumsum(y)

        # decimator

        y = y[::r]
        # comb stage
        return np.diff(y, n=n, prepend=np.zeros(n))

    def interpolator(self, r, n, mode=False):
        """
        CIC inteprolator: Comb + Decimator + Integrator

        Parameters
        ----------
        r : int
            interpolation rate
        n : int
            filter order
        mode : bool
            False - zero padding, True - value padding.
        """

        # comb stage
        y = np.diff(self.x, n=n,
                    prepend=np.zeros(n), append=np.zeros(n))

        # interpolation
        if mode:
            y = np.repeat(y, r)
        else:
            y = np.array([i if j == 0 else 0 for i in y for j in range(r)])

        # integrator
        for i in range(n):
            y = np.cumsum(y)

        if mode:
            return y[1:1 - n * r]
        else:
            return y[r - 1:-n * r + r - 1]


sample_rate = 100.0
nsamples = 400
ampl = 15
freq = 5
noiseCount = 100

t = np.arange(nsamples) / sample_rate
# x = cos(2*pi*0.5*t) + 0.2*sin(2*pi*2.5*t+0.1) + \
#         0.2*sin(2*pi*15.3*t) + 0.1*sin(2*pi*16.7*t + 0.1) + \
#             0.1*sin(2*pi*23.45*t+.8)


x = ampl * np.sin(2 * np.pi * freq * t)
x = func.addSomeNoise(x, noiseCount, t, freq, ampl)

decimation = 4
interpolation = 1
filterOrder = 4

cicFilter = CicFilter(x)
# decRes = cicFilter.decimator(decimation, filterSize)
# intRes = cicFilter.interpolator(interpolation, filterSize, True)

filtered = cicFilter.decimator(decimation, filterOrder)
# filtered = cicFilter.interpolator(decimation, filterSize, True)

plt.figure()

plt.subplot(2, 1, 1)
plt.title('Signal')
plt.plot(x, 'g')
plt.grid()

plt.subplot(2, 1, 2)
plt.title('Filtered')
plt.plot(filtered, 'b')
plt.grid()

# plt.figure()

# plt.subplot(2, 1, 1)
# plt.title('decRes')
# plt.plot(decRes, 'r')
# plt.grid()

# plt.subplot(2, 1, 2)
# plt.title('intRes')
# plt.plot(intRes, 'r')
# plt.grid()

plt.show()
