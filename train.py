
#!/usr/bin/python
#
# training file for i3d model.
# this project supports the udacity self-driving dataset.
# link: https://github.com/udacity/self-driving-car/tree/master/datasets
# also supports the comma ai dataset
#
# Author: Neil (Yongyang) Nie
# Copyright: (c) 2018
# Licence: MIT
# Contact: contact@neilnie.com
#

from i3d import i3d
import configs
import helper
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    i3d_model = i3d(input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, configs.CHANNELS) , weights_path='./i3d_speed_c_64_10.h5')

    labels = pd.read_csv('/home/neil/dataset/speedchallenge/data/data.csv').values
    val_label = pd.read_csv('/home/neil/dataset/speedchallenge/data/validation.csv').values

    train_gen = helper.comma_batch_generator(batch_size=1, data=labels, augment=True)
    val_gen = helper.comma_validation_generator(batch_size=1, data=val_label)

    # imgs, angles = next(train_gen)
    # for b in imgs:
    #     for im in b:
    #         plt.imshow(im)
    #         plt.show()

    i3d_model.train(train_gen=train_gen, epochs=5, epoch_steps=3000, validation=True, val_gen=val_gen, val_steps=2000, save_path='./i3d_speed_c_64_11.h5')


