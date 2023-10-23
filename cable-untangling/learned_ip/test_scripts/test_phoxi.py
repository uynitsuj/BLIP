
import matplotlib.pyplot as plt

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
ax1.set_title("Intensity Image")
ax1.imshow(frame.color.data)
ax1.axis("off")
ax2.set_title("Depth Image")
ax2.imshow(frame.depth.data)
ax2.axis("off")
plt.show()