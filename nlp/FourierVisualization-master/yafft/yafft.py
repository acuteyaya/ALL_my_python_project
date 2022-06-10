import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from numpy import fft
import scipy.io.wavfile as wf

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus']=False
y, sr = librosa.load('cache.wav', sr=20000)
print(y.shape,type(y))

x=np.linspace(0,1,1200)
y=7*np.sin(2*np.pi*200*x) + 5*np.sin(2*np.pi*400*x)+6*np.sin(2*np.pi*600*x)
fft_y=np.fft.fft(y)                          #快速傅里叶变换

fft_y[1:500]=0+0j
fft_y[700:1200]=0+0j
N = len(fft_y)
N1=N
#N1=N//2
#fft_y=fft_y[0:N1]
plt.plot(abs(fft_y))
plt.title("降噪后的频率振幅谱")
plt.xlabel("频率")
plt.ylabel("amplitude")
plt.show()
data=np.fft.ifft(fft_y)
plt.plot(np.arange(N), data)
plt.title('降噪波形')
plt.show()


x = np.arange(N1)
abs_y = np.abs(fft_y)
angle_y = np.angle(fft_y)
normalization_y = np.array(abs_y / N1).astype(np.float64)

plt.subplot(221)
plt.plot(np.arange(N), y)
plt.title('原始波形')
plt.subplot(222)
plt.plot(x, fft_y, 'black')
plt.title('振幅谱(未求振幅绝对值)', fontsize=9, color='black')
plt.subplot(223)
plt.plot(x, abs_y, 'r')
plt.title('振幅谱(未归一化)', fontsize=9, color='red')
plt.subplot(224)
plt.plot(x, angle_y, 'violet')
plt.title('相位谱(未归一化)', fontsize=9, color='violet')
plt.show()
plt.plot(x, normalization_y )
plt.title('归一化')
plt.show()

n0 = 0
n1 = N
print(N)
plt.figure(figsize=(14, 5))
plt.plot(y[n0:n1])
plt.grid()
#plt.show()
zero_crossings = librosa.zero_crossings(y[n0:n1], pad=False)
print(sum(zero_crossings))
