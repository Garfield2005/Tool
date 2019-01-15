#!/usr/bin/python

import os
import numpy as np
from PIL import Image
import cv2

video_file = "/data/deeplearning/zhangmm/segmentation/video/source.mp4"
frame_dir = "/data/deeplearning/zhangmm/segmentation/images"
interval = 5

frame_dir = os.path.join(frame_dir, os.path.basename(video_file))
if not os.path.exists(frame_dir):
    os.mkdir(frame_dir)

def video2frame(video_file, frame_dir, interval):
    cap = cv2.VideoCapture(video_file)
    success = cap.isOpened()
    if not success:
        print("failed")
        return None

    frames = []
    frame_index = 0
    while(success):
        success, frame = cap.read()
        frame_index += 1
        print(">>>>> reading %d frame:" % frame_index)
        if not success:
            print("get %d frame failed:" % frame_index)
            continue
        if frame_index == 1 or frame_index % interval == 0:
            frame_path = os.path.join(frame_dir, "{}.jpg".format(frame_index))
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)

    cap.release()
    return frames

frames = video2frame(video_file, frame_dir, interval)
print("get frames: {}".format(len(frames)))
