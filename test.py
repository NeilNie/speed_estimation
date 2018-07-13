
#
#

from i3d import i3d
import configs
import helper
import pandas as pd

if __name__ == '__main__':

    i3d_model = i3d(input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, configs.CHANNELS))
    i3d_model.summary()

    labels = pd.read_csv('/home/neil/dataset/udacity/main.csv').values
    val_label = pd.read_csv('/home/neil/dataset/steering/test/labels.csv')
    train_gen = helper.udacity_batch_generator(batch_size=1, data=labels, augment=False)
    val_gen = helper.validation_batch_generator(batch_size=1, data=labels)

    i3d_model.train(train_gen=train_gen, val_gen=val_gen)