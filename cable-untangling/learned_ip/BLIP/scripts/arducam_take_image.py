import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

current_time = datetime.now().strftime('%Y%m%d%H%M%S')
# Open the video device
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Couldn't open the video device!")
    exit()

# Capture a single frame
ret, img = cap.read()

if ret:
    # Save the captured frame as an image
    cv2.imwrite('./calibration/image'+current_time+'.jpg', img)
    print("Saved to " + './calibration/image'+current_time+'.jpg')
else:
    print("Error: Couldn't capture an image!")

# Release the video device
cap.release()
