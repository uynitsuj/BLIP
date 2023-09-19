from matplotlib import pyplot as plt
import cv2
import numpy as np
data = np.load('oct_11_overhand/color_1.npy')
# plt.savefig()
# plt.imshow(data, cmap='gray')
# plt.show()

cv2.imwrite('oct_11_overhand/color_1_img.png', data)