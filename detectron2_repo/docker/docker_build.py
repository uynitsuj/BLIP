#!/usr/bin/env python
import os

#if __name__=="__main__":
#   cmd = "docker build -t detectron2:v0 ."
#   code = os.system(cmd)

os.system("docker build --build-arg USER_ID=$UID -t detectron2:v0 .")
