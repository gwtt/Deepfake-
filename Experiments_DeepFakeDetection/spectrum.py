import numpy as np
import cv2
from matplotlib import pyplot as plt

filename = './'
img = cv2.imread(filename)  # 以灰度模式读取图像
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
f = np.fft.fft2(img)  # 进行二维快速傅里叶变换（FFT），得到频域表示 f
fshift = np.fft.fftshift(f)  # 对 FFT 结果进行频谱中心化，将零频分量移动到频谱的中心。
magnitude_spectrum = 20 * np.log(np.abs(fshift))  # 计算幅度谱，并取其对数得到 magnitude_spectrum
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

