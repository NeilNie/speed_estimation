"""
preprocess dataset
Neil Nie (c) 2018, All Rights Reserved
Contact: contact@neilnie.com

"""

import cv2
import numpy as np
import pandas as pd
import time


def video2frames():

    vidcap = cv2.VideoCapture('/Users/yongyangnie/Desktop/speedchallenge/data/train.mp4')
    success, image = vidcap.read()
    count = 0

    while success:
        cv2.imwrite("/Users/yongyangnie/Desktop/speedchallenge/data/frames/train/frame%d.jpg" % count,
                    image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1

        if count % 1000 == 0:
            print('Read a new frame: ', success)


def process_optical_flow():

    labels = pd.read_csv("/home/neil/dataset/speedchallenge/data/data.csv").values
    tvl1 = cv2.DualTVL1OpticalFlow_create()
    c = 0
    for i in range(0, len(labels)-2):

        path_previous = "/home/neil/dataset/speedchallenge/data/train/" + str(labels[i][1])
        path_next = "/home/neil/dataset/speedchallenge/data/train/" + str(labels[i+1][1])

        prev_frame_gray = cv2.cvtColor(cv2.imread(path_previous), cv2.COLOR_BGR2GRAY)
        next_frame_gray = cv2.cvtColor(cv2.imread(path_next), cv2.COLOR_BGR2GRAY)

        flow = tvl1.calc(prev_frame_gray, next_frame_gray, None)

        np_mat = np.array(flow)

        np.save(file="/home/neil/dataset/speedchallenge/data/train/flow/frame" + str(i), arr=np_mat)
        c+=1
        if c == 100:
            exit(0)


if __name__ == '__main__':

    process_optical_flow()