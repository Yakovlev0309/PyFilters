import signal_gen as sig
import matplotlib.pyplot as mpl
import fir_filter as fir
import numpy as np

ampl = 1
begin = 0
end = 100
rate = 1
signal = sig.getComplexSin(ampl, begin, end, rate)

x = signal[0]
real = signal[1]
imag = signal[2]

print(imag)

mpl.plot(x, real, 'g', label="real")
mpl.plot(x, imag, 'r', label="imag")

fReal = fir.filter(real)
# fReal = fir.filter2(real, real, real.size)
mpl.plot(fReal, 'b', label="filtered")

mpl.legend()
mpl.show()

