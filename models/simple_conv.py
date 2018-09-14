"""Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.

The model is introduced in:

Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
Joao Carreira, Andrew Zisserman
https://arxiv.org/abs/1705.07750v1

Initially written by Ese dlpbc
Modified & improved by Neil Nie.

MIT Licence. (c) Yongyang Nie, 2018 All Rights Reserved
Contact: contact@neilnie.com

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import ELU
from keras.optimizers import SGD
from keras.layers import Flatten

from keras.models import load_model
from keras.callbacks import TensorBoard
import datetime
import pandas as pd
import helper
import configs


class ConvModel:

    def __init__(self, input_shape, weights_path=None, classes=1):


        '''Instantiates the Inflated 3D Inception v1 architecture.

        Optionally loads weights pre-trained on Kinetics. Note that when using TensorFlow,
        Always channel last. The model and the weights are compatible with both
        TensorFlow. The data format convention used by the model is the one
        specified in your Keras config file.
        Note that the default input frame(image) size for this model is 224x224.

        :param weights_path: one of `None` (random initialization)
        :param input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape should have exactly
                3 inputs channels. NUM_FRAMES should be no smaller than 8. The authors
                used 64 frames per example for training and testing on kinetics dataset
                Width and height should be no smaller than 32.
                i.e.: `(64, 150, 150, 3)` would be one valid value.
        :param dropout_prob: optional, dropout probability applied in dropout layer
                after global average pooling layer.
                0.0 means no dropout is applied, 1.0 means dropout is applied to all features.
                Note: Since Dropout is applied just before the classification
                layer, it is only useful when `include_top` is set to True.
        :param endpoint_logit: (boolean) optional. If True, the model's forward pass
                will end at producing logits. Otherwise, softmax is applied after producing
                the logits to produce the class probabilities prediction. Setting this parameter
                to True is particularly useful when you want to combine results of rgb model
                and optical flow model.
                - `True` end model forward pass at logit output
                - `False` go further after logit to produce softmax predictions
                Note: This parameter is only useful when `include_top` is set to True.
        :param classes: For regression (i.e. behavorial cloning) 1 is the default value.
                optional number of classes to classify images into, only to be specified
                if `include_top` is True, and if no `weights` argument is specified.

        '''
        self.classes = classes
        self.weight_path = weights_path

        img_input = Input(shape=input_shape)
        self.model = self.commaai_model()

        if weights_path:
            self.model = load_model(weights_path)
            print("loaded weights:" + weights_path)

    def summary(self):
        print(self.model.summary())

    def train(self, type, epochs=10, epoch_steps=5000, val_steps=None, validation=False, log_path="logs", save_path=None):

        '''training the model

        :param type: tye type of model. Choices are: flow or rgb
        :param train_gen: training generator. For details, please read the
        implementation in helper.py
        :param val_gen: validation generator, for now it's required.
        :param epoch: number of training epochs.
        :param epoch_steps: number of training steps per epoch. (!= batch_size)
        :param val_steps: number of validation steps
        :param log_path: training log path.
        :param validation: run validation or not. If not validating, val_gen and val_steps can be non.
        '''

        labels = pd.read_csv('/home/neil/dataset/speedchallenge/data/data.csv').values
        val_label = pd.read_csv('/home/neil/dataset/speedchallenge/data/validation.csv').values

        if type == 'flow':
            train_gen = helper.comma_flow_rgb_batch_gen(batch_size=8, data=labels)
            val_gen = helper.comma_flow_rgb_batch_gen(batch_size=8, data=val_label)
        # elif type == 'rgb':
        #     train_gen = helper.comma_batch_generator(batch_size=1, data=labels, augment=True)
        #     val_gen = helper.comma_validation_generator(batch_size=1, data=val_label)
        else:
            raise Exception('Sorry, the model type is not recognized')

        if save_path is None:
            print("[WARNING]: trained model will not be saved. Please specify save_path")

        tensorboard = TensorBoard(log_dir=(log_path + "/conv/{}".format(datetime.datetime.now())))

        if validation:
            if val_steps:
                self.model.fit_generator(train_gen, steps_per_epoch=epoch_steps,
                                         epochs=epochs, validation_data=val_gen, validation_steps=val_steps,
                                         verbose=1, callbacks=[tensorboard])  #
            else:
                raise Exception('please specify val_steps')

        else:
            self.model.fit_generator(train_gen, steps_per_epoch=epoch_steps,
                                     epochs=epochs, verbose=1, callbacks=[tensorboard])

        self.model.save(save_path)

    def commaai_model(self):

        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(configs.IMG_HEIGHT, configs.IMG_WIDTH, 3), output_shape=(configs.IMG_HEIGHT, configs.IMG_WIDTH, 3)))
        model.add(Conv2D(16, (8, 8), strides=4, padding="same"))
        model.add(ELU())
        model.add(Conv2D(32, (5, 5), strides=2, padding="same"))
        model.add(ELU())
        model.add(Conv2D(64, (5, 5), strides=2, padding="same"))
        model.add(Flatten())
        model.add(Dropout(.2))
        model.add(ELU())
        model.add(Dense(512))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(1))

        sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='mean_squared_error')
        # print('steering model is created and compiled...')
        return model
