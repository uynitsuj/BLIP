#!/usr/bin/env python
import os

# os.system("nvidia-docker run -it -v /home/vainavi/detectron2/:/host detectron2:v0")

if __name__=="__main__":
    cmd = "nvidia-docker run -it -v %s:/host \
                                 -v %s:/host/data detectron2:v0" % (os.path.join(os.getcwd(), '..'), '/raid/vainavi/data/long_cable')
    #cmd = "nvidia-docker run -it priya-keypoints"
    #cmd = "docker run --runtime=nvidia -it -v %s:/host priya-keypoints" % (os.path.join(os.getcwd(), '..'))
    code = os.system(cmd)
