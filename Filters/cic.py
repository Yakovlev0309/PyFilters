class integrator:
	def __init__(self):
		self.yn  = 0
		self.ynm = 0
	
	def update(self, inp):
		self.ynm = self.yn
		self.yn  = (self.ynm + inp)
		return (self.yn)
		
class comb:
	def __init__(self):
		self.xn  = 0
		self.xnm = 0
	
	def update(self, inp):
		self.xnm = self.xn
		self.xn  = inp
		return (self.xn - self.xnm)


import matplotlib.pyplot as plt
import numpy as np
import functions as func


sample_rate = 100.0
nsamples = 400
ampl = 15
freq = 5
noiseCount = 100

t = np.arange(nsamples) / sample_rate
signal = ampl * np.sin(2 * np.pi * freq * t)
signal = func.addSomeNoise(signal, noiseCount, t, freq, ampl)

decimation = 2		# any integer; powers of 2 work best.
stages = 2			# pipelined I and C stages

## Calculate normalising gain
gain = (decimation * 1) ** stages

## Seperate Stages - these should be the same unless you specifically want otherwise.
c_stages = stages
i_stages = stages

## Generate Integrator and Comb lists (Python list of objects)
intes = [integrator() for a in range(i_stages)]
combs = [comb()	      for a in range(c_stages)]

## Decimating CIC Filter
filtered = []
for (s, v) in enumerate(signal):
	z = v
	for i in range(i_stages):
		z = intes[i].update(z)
	
	if (s % decimation) == 0: # decimate is done here
		for c in range(c_stages):
			z = combs[c].update(z)
			j = z
		filtered.append(j / gain) # normalise the gain

## Crude function to FFT and slice data, with 20log10 result
def fft_this(data):
	N = len(data)
	return (20*np.log10(np.abs(np.fft.fft(data)) / N)[:N // 2])


## Plot some graphs
plt.figure(1)
plt.suptitle("Simple Test of Decimating CIC filter")
plt.subplot(2,2,1)
plt.title("Time domain input")
plt.plot(signal)
plt.grid()
plt.subplot(2,2,3)
plt.title("Frequency domain input")
plt.plot(fft_this(signal))
plt.grid()
plt.subplot(2,2,2)
plt.title("Time domain output")
plt.plot(filtered)
plt.grid()
plt.subplot(2,2,4)
plt.title("Frequency domain output")
plt.plot(fft_this(filtered))
plt.grid()

## Try to calculate the frequency rolloff. Just for indication!
## These much match signals in the "inp_samp()" function.
fos = fft_this(filtered)
try:
	f = 40
	f2 = f * 10
	print("Filtered Output, bin %4d = %f" % (f,  fos[f]))
	print("Filtered Output, bin %4d = %f" % (f2, fos[f2]))
	print("Difference %f in a decade" % (fos[40] - fos[400]))
except:
	print("*** Error: Cannot FFT bins must be chosen to match decimation and frequencies in the inp_samp() function.")
	pass
## Show graphs
plt.show()