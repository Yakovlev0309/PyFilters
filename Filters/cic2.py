import numpy as np
import functions as func
import matplotlib.pyplot as plt
from cic_functions import CicFilter


sample_rate = 100.0
nsamples = 400
ampl = 15
freq = 5
noiseCount = 100

t = np.arange(nsamples) / sample_rate
# x = cos(2*pi*0.5*t) + 0.2*sin(2*pi*2.5*t+0.1) + \
#         0.2*sin(2*pi*15.3*t) + 0.1*sin(2*pi*16.7*t + 0.1) + \
#             0.1*sin(2*pi*23.45*t+.8)
t[0] = sample_rate

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
