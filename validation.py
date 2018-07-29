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
import matplotlib.pyplot as plt


def validation_score(model_path, type, save=False, debugging=False):

    model = i3d(input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, configs.CHANNELS),
                    weights_path=model_path)

    # read the steering labels and image path
    df_truth = pd.read_csv('/home/neil/dataset/speedchallenge/data/validation.csv').values

    esum = 0
    input = []
    predictions = []

    if type == 'flow':

        raise Exception('Sorry, the model type is not supported')

    elif type == 'rgb':

        start_time = time.time()

        for i in range(configs.LENGTH):
            file = configs.TRAIN_DIR + str(df_truth[i][1])
            img = helper.load_image(file)
            input.append(img)

        # Run through all images
        for i in range(configs.LENGTH, len(df_truth)):

            p = configs.TRAIN_DIR + str(df_truth[i][1])
            img = helper.load_image(p)
            input.pop(0)
            input.append(img)
            input_array = np.array([np.asarray(input)])
            prediction = model.model.predict(input_array)[0][0]
            actual_steers = df_truth[i][2]
            e = (actual_steers - prediction) ** 2
            esum += e

            predictions.append(prediction)

            if len(predictions) % 1000 == 0:
                print('.')

    elif type == 'rgb-flow':

        print('Started')

        start_time = time.time()

        previous = helper.load_image(configs.TRAIN_DIR + str(df_truth[0][1]))

        for i in range(1, configs.LENGTH + 1):

            img = helper.load_image(configs.TRAIN_DIR + str(df_truth[i][1]))
            rgbImg = helper.optical_flow_rgb(previous=previous, current=img)
            if debugging:
                # fig = plt.figure(figsize=(3, 1))
                # fig.add_subplot(1, 1, 1)
                plt.imshow(rgbImg)
                # fig.add_subplot(1, 2, 1)
                plt.imshow(previous)
                # fig.add_subplot(1, 3, 1)
                plt.imshow(img)
                plt.show()

            previous = img
            input.append(rgbImg)

        previous = helper.load_image(configs.TRAIN_DIR + str(df_truth[configs.LENGTH + 1][1]))

        for i in range(configs.LENGTH + 2, len(df_truth)):

            img = helper.load_image(configs.TRAIN_DIR + str(df_truth[i][1]))
            rgb_flow = helper.optical_flow_rgb(previous, img)
            if debugging:
                plt.imshow(rgb_flow)
                plt.show()
            input.pop(0)
            input.append(rgb_flow)
            input_array = np.array([np.asarray(input)])
            prediction = model.model.predict(input_array)[0][0]
            actual_steers = df_truth[i][2]
            e = (actual_steers - prediction) ** 2
            esum += e

            predictions.append(prediction)

            if len(predictions) % 1000 == 0:
                print('.')
    else:
        raise Exception('Sorry, the model type is not recognized')

    print("time per step: %s seconds" % ((time.time() - start_time) / len(predictions)))

    if save:
        print("Writing predictions...")
        pd.DataFrame({"steering_angle": predictions}).to_csv('./result.csv', index=False, header=True)
        print("Done!")

    return esum / len(predictions)


if __name__ == "__main__":

    print("Validating...")
    score = validation_score('./i3d_speed_comma_multiflow_32_4.h5', type='rgb-flow', debugging=False)
    print("Finished!")
    print(score)

