
#!/usr/bin/python
#
# training file for i3d model.
# this project supports the udacity self-driving dataset.
# link: https://github.com/udacity/self-driving-car/tree/master/datasets
#
# will soon support Common AI speed dataset.
#
# Author: Neil (Yongyang) Nie
# Copyright: (c) 2018
# Licence: MIT
# Contact: contact@neilnie.com
#


import helper
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    labels = pd.read_csv('/home/neil/dataset/speedchallenge/data/data.csv').values

    train_gen = helper.comma_flow_batch_generator(batch_size=1, data=labels)

    imgs, angles = next(train_gen)
    print(imgs.shape)

    for b in imgs:
        for im in b:
            plt.imshow(im[:, :, 0])
            plt.show()

