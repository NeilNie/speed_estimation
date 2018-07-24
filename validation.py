#
# (c) Yongyang Nie
# 2018, All Rights Reserved
#


from i3d import i3d
import cv2
import configs
from os import path
import pandas as pd
import numpy as np
import helper
import math
import time


def validation_score(model_path):

    model = i3d(input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, configs.CHANNELS),
                    weights_path=model_path)

    # read the steering labels and image path
    df_truth = pd.read_csv('/home/neil/dataset/speedchallenge/data/validation.csv').values

    esum = 0
    count = 0
    input = []
    predictions = []

    start_time = time.time()

    for i in range(configs.LENGTH):
        file = "/home/neil/dataset/speedchallenge/data/train/" + str(df_truth[i][1])
        img = helper.load_image(file)
        input.append(img)

    # Run through all images
    for i in range(configs.LENGTH, len(df_truth)):

        p = "/home/neil/dataset/speedchallenge/data/train/" + str(df_truth[i][1])
        img = helper.load_image(p)
        input.pop(0)
        input.append(img)
        input_array = np.array([np.asarray(input)])
        prediction = model.model.predict(input_array)[0][0]
        actual_steers = df_truth[i][2]
        e = (actual_steers - prediction) ** 2
        esum += e
        count += 1

        predictions.append(prediction)

        if count % 1000 == 0:
            print('.')

    print("time per step: %s seconds" % ((time.time() - start_time) / len(predictions)))
    # print("Writing predictions...")
    # pd.DataFrame({"steering_angle": predictions}).to_csv('./result.csv', index=False, header=True)
    # print("Done!")

    return math.sqrt(esum / len(predictions))


if __name__ == "__main__":

    print("Validating...")
    score = validation_score('./i3d_speed_c_64_7.h5')
    print("Finished!")
    print(score)

    # score2 = validation_score('./i3d_speed_c_64_8.h5')
    # print(score2)