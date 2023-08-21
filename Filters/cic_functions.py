import functions as func
from functions import np


def maf_conv(signal, m=2):
    """
    Calculate moving average filter via convolution
    Parameters
    ----------
    m : int
        moving average step
    """
    coe = np.ones(m) / m
    return np.convolve(signal, coe, mode='same')    


def maf(signal, sample_rate, m=2):
    """
    Calculate moving average filter via convolution
    Parameters
    ----------
    m : int
        moving average step
    """
    coe = func.compute_fir_filter_coefficients(m, 10.0, sample_rate)
    return func.convolution(signal, coe)


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
