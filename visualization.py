"""
Dataset visualization tool
Original By: Comma.ai and Chris Gundling
Revised and used by Neil Nie
"""

import matplotlib.backends.backend_agg as agg
import numpy as np
import pandas as pd
import cv2
import pygame
import os
import pylab
from i3d import Inception3D
import helper
import configs
from termcolor import colored

pygame.init()
size = (640, 640)
pygame.display.set_caption("speed prediction viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
screen.set_alpha(None)

camera_surface = pygame.surface.Surface((640, 480), 0, 24).convert()
clock = pygame.time.Clock()

PI_RAD = (180 / np.pi)
white = (255, 255, 255)
red = (255, 102, 102)
blue = (102, 178, 255)
black = (0, 0, 0)

# Create second screen with matplotlibs
fig = pylab.figure(figsize=[6.4, 1.6], dpi=100)
ax = fig.gca()
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
line1, = ax.plot([], [], 'b.-', label='Model')
line2, = ax.plot([], [], 'r.-', label='Human')
a = []
b = []
ax.legend(loc='upper left', fontsize=8)

myFont = pygame.font.SysFont("monospace", 18)
static_label1 = myFont.render('Model Prediction:', 1, white)
static_label2 = myFont.render('Ground Truth:', 1, white)
static_label3 = myFont.render('Abs. Error', 1, black)

def test_loop(model_path, model_type):

    """
    for visualizing acceleration models with the comma AI
    validation dataset. The speed is initialized before
    acceleration prediction starts.

    :param model_path: the path of the trained Keras model
    :param model_type: the type of model, rgb, flow or rgb-flow
    :return: None
    """

    print(colored('Preparing', 'blue'))

    model = Inception3D(weights_path=model_path, input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, 2))

    # read the steering labels and image path
    files = os.listdir(configs.TEST_DIR)

    inputs = []
    starting_index = 9000
    end_index = 1000

    if model_type == 'rgb':

        for i in range(starting_index, starting_index + configs.LENGTH):
            img = helper.load_image(configs.TEST_DIR + "frame" + str(i) + ".jpg")
            inputs.append(img)

        print(colored('Started', 'blue'))

        # Run through all images
        for i in range(starting_index + configs.LENGTH + 1, len(files) - 1 - end_index):

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

            img = helper.load_image(configs.TEST_DIR + "frame" + str(i) + ".jpg", resize=False)
            in_frame = cv2.resize(img, (configs.IMG_WIDTH, configs.IMG_HEIGHT))
            inputs.pop(0)
            inputs.append(in_frame)
            prediction = model.model.predict(np.array([inputs]))[0][0]

            pygame_loop(label=0, prediction=prediction, img=img)

    elif model_type == 'flow':

        previous = helper.load_image(configs.TEST_DIR + "frame" + str(starting_index) + ".jpg")

        for i in range(starting_index, starting_index + configs.LENGTH):

            img = helper.load_image(configs.TEST_DIR + "frame" + str(i) + ".jpg")
            in_frame = cv2.resize(img, (configs.IMG_WIDTH, configs.IMG_HEIGHT))
            flow = helper.optical_flow(previous=previous, current=in_frame)
            inputs.append(flow)

        previous = helper.load_image(configs.TEST_DIR + "frame" + str(starting_index + configs.LENGTH) + ".jpg")

        for i in range(starting_index + configs.LENGTH + 1, len(files) - 1 - end_index):

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

            img = helper.load_image(configs.TEST_DIR + "frame" + str(i) + ".jpg", resize=False)
            in_frame = cv2.resize(img, (configs.IMG_WIDTH, configs.IMG_HEIGHT))
            flow = helper.optical_flow(previous, in_frame)
            inputs.pop(0)
            inputs.append(flow)
            input_array = np.array([np.asarray(inputs)])
            prediction = model.model.predict(input_array)[0][0]

            pygame_loop(label=0, prediction=prediction, img=img)

    else:
        raise Exception('Sorry, the model type is not recognized')


def pygame_loop(label, prediction, img):

    if prediction < 0:
        pred_label = myFont.render('Slowing', 1, white)
    else:
        pred_label = myFont.render('Speeding', 1, white)
    gt_label = myFont.render(str(label), 1, red)
    pred_val = myFont.render(str(prediction), 1, blue)
    error_label = myFont.render(str(abs(round((prediction - label), 3))), 1, black)

    a.append(prediction)    # a is prediction
    b.append(label)         # b is label
    line1.set_ydata(a)
    line1.set_xdata(range(len(a)))
    line2.set_ydata(b)
    line2.set_xdata(range(len(b)))
    ax.relim()
    ax.autoscale_view()

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    surf = pygame.image.fromstring(raw_data, size, "RGB")
    screen.blit(surf, (0, 480))

    # draw on
    pygame.surfarray.blit_array(camera_surface, img.swapaxes(0, 1))
    screen.blit(camera_surface, (0, 0))
    screen.blit(static_label3, (15, 15))
    screen.blit(error_label, (15, 30))
    screen.blit(static_label1, (50, 420))
    screen.blit(pred_label, (280, 420))
    screen.blit(pred_val, (50, 450))

    screen.blit(static_label2, (450, 420))
    screen.blit(gt_label, (450, 450))

    clock.tick(60)
    pygame.display.flip()


def visualize_accel(model_path, label_path, model_type):

    print(colored('Preparing', 'blue'))

    model = Inception3D(weights_path=model_path, input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, 3))

    # read the steering labels and image path
    labels = pd.read_csv(label_path).values

    inputs = []
    starting_index = 1
    end_index = 0
    # init_speed = labels[starting_index][2]

    if model_type == 'rgb':

        for i in range(starting_index, starting_index + configs.LENGTH):
            img = helper.load_image(configs.TEST_DIR + "frame" + str(i) + ".jpg")
            inputs.append(img)

        print(colored('Started', 'blue'))

        # Run through all images
        for i in range(starting_index + configs.LENGTH + 1, len(labels) - 1 - end_index):

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

            img = helper.load_image("/home/neil/dataset/speedchallenge/data/train/" + str(labels[i][1]), resize=False)
            in_frame = cv2.resize(img, (configs.IMG_WIDTH, configs.IMG_HEIGHT))
            inputs.pop(0)
            inputs.append(in_frame)
            pred_accel = model.model.predict(np.array([np.asarray(inputs)]))[0][0]
            label_accel = (labels[i][2] - labels[i - 1][2]) * 20
            pygame_loop(label=label_accel, prediction=pred_accel, img=img)

    elif model_type == 'flow':

        previous = helper.load_image("/home/neil/dataset/speedchallenge/data/train/" + str(labels[i][1]))

        for i in range(starting_index, starting_index + configs.LENGTH):
            img = helper.load_image(configs.TEST_DIR + "frame" + str(i) + ".jpg")
            in_frame = cv2.resize(img, (configs.IMG_WIDTH, configs.IMG_HEIGHT))
            flow = helper.optical_flow(previous=previous, current=in_frame)
            inputs.append(flow)

        previous = helper.load_image("/home/neil/dataset/speedchallenge/data/train/" + str(labels[i][1]))

        for i in range(starting_index + configs.LENGTH + 1, len(labels) - 1 - end_index):

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

            img = helper.load_image("/home/neil/dataset/speedchallenge/data/train/" + str(labels[i][1]))
            in_frame = cv2.resize(img, (configs.IMG_WIDTH, configs.IMG_HEIGHT))
            flow = helper.optical_flow(previous, in_frame)
            inputs.pop(0)
            inputs.append(flow)
            pred_accel = model.model.predict(np.array([np.asarray(inputs)]))[0][0]
            label_accel = (labels[i][2] - labels[i-1][2]) * 20
            pygame_loop(label=label_accel, prediction=pred_accel, img=img)
    else:
        raise Exception('Sorry, the model type is not recognized')


if __name__ == "__main__":

    # test_loop(model_path='i3d_speed_comma_flow_32_9.h5', model_type='flow')
    visualize_accel(label_path='/home/neil/dataset/speedchallenge/data/validation.csv', model_path='i3d_accel_rgb_32_4.h5', model_type='rgb')
