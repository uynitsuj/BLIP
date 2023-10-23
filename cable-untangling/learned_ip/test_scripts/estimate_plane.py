
import matplotlib.pyplot as plt
import numpy as np
import random

from phoxipy import PhoXiSensor

p = PhoXiSensor("1703005")
p.start()

# for _ in range(5):
#     frame = p.read()
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))
#     ax1.set_title("Intensity Image")
#     ax1.imshow(frame.color.data)
#     ax1.axis("off")
#     ax2.set_title("Depth Image")
#     ax2.imshow(frame.depth.data)
#     ax2.axis("off")
#     plt.show()

# p.stop()

frame = p.read()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))

x_lims = (250, 850)
y_lims = (300, 600)

# sample n times for points
n = 20

data = np.ones((n, 3))
height = np.zeros((n, 1))

for i in range(n):
    x_choice = random.randint(*x_lims)
    y_choice = random.randint(*y_lims)
    data[i] = np.array([x_choice, y_choice, 1])
    height[i] = frame.depth.data[y_choice][x_choice]

print(data)

print(np.linalg.lstsq(data, height))
