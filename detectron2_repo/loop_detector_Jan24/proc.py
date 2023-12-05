import cv2
import numpy as np
import glob
imgs=glob.glob("color_*.npy")
print(imgs)
N=30
for i in range(323,323+N):
    im=np.load(f"color_{i}.npy")
    cv2.imwrite(f"color_{i}.png",im)
