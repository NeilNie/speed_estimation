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
from simple_conv import ConvModel

if __name__ == '__main__':

    i3d_model = i3d(input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, 2)) # weights_path='./i3d_speed_c_64_10.h5'
    i3d_model.summary()
    i3d_model.train(type='rgb-flow', epochs=5, epoch_steps=3000, validation=True, val_steps=2000, save_path='./i3d_speed_comma_flow_64_0.h5')

    # model = ConvModel(input_shape=(configs.IMG_HEIGHT, configs.IMG_WIDTH, 3), weights_path='./conv_speed_comma_frgb_1.h5')
    # model.train(type='flow', epochs=2, epoch_steps=1000, validation=True, val_steps=200, save_path='./conv_speed_comma_frgb_2.h5')
