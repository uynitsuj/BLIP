import numpy as np
import matplotlib.pyplot as plt 
import cv2



data = np.load("config2/color_21.npy")
cv2.imwrite('config_images/config2_21.png', data)
