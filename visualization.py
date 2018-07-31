'''
Dataset visualization tool
Original By: Comma.ai and Chris Gundling
Revised and used by Neil Nie
'''

from __future__ import print_function
import numpy as np
import cv2
import pygame
import pandas as pd
from os import path
import matplotlib
import matplotlib.backends.backend_agg as agg
import os
import pylab
from i3d import i3d
import helper
import configs
import time

pygame.init()
size = (640, 650)
pygame.display.set_caption("speed prediction viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
screen.set_alpha(None)

camera_surface = pygame.surface.Surface((640, 480), 0, 24).convert()
clock = pygame.time.Clock()

PI_RAD = (180 / np.pi)
red = (255, 0, 0)
blue = (0, 0, 255)


def test_loop(model_path, model_type):

    '''
    for visualizing the model with the comma AI
    test dataset. The ds doesn't contain training labels.
    '''

    # ------------------------------------------------
    model = i3d(weights_path=model_path, input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, configs.CHANNELS))

    # read the steering labels and image path
    TEST_DIR = '/home/neil/dataset/speedchallenge/data/test'
    files = os.listdir(TEST_DIR)

    # Create second screen with matplotlibs
    fig = pylab.figure(figsize=[6.4, 1.6], dpi=100)
    ax = fig.gca()
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    line1, = ax.plot([], [], 'b.-', label='Human')
    A = []
    ax.legend(loc='upper left', fontsize=8)

    myFont = pygame.font.SysFont("monospace", 18)
    randNumLabel = myFont.render('Human Driving Speed:', 1, blue)

    input = []
    starting_index = 8000

    if model_type == 'rgb':

        for i in range(starting_index, starting_index + configs.LENGTH):
            img = helper.load_image(TEST_DIR + "frame" + str(i) + ".jpg")
            input.append(img)

        # Run through all images
        for i in range(starting_index + configs.LENGTH + 1, len(files) - 1):

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

            img = helper.load_image(TEST_DIR + "frame" + str(i) + ".jpg", resize=False)
            in_frame = cv2.resize(img, (configs.IMG_WIDTH, configs.IMG_HEIGHT))
            input.pop(0)
            input.append(in_frame)
            input_array = np.array([input])
            prediction = model.model.predict(input_array)[0][0]

    elif model_type == 'flow':

        previous = helper.load_image(TEST_DIR + "frame" + str(starting_index) + ".jpg", resize=False)

        for i in range(starting_index, starting_index + configs.LENGTH):

            img = helper.load_image(TEST_DIR + "frame" + str(i) + ".jpg")
            flow = helper.optical_flow(previous=previous, current=img)
            input.append(flow)

        previous = helper.load_image(TEST_DIR + "frame" + str(starting_index + configs.LENGTH) + ".jpg", resize=False)

        for i in range(starting_index + configs.LENGTH + 1, len(files) - 1):

            img = helper.load_image(TEST_DIR + "frame" + str(i) + ".jpg", resize=False)
            # TODO:
            flow = helper.optical_flow(previous, img)
            input.pop(0)
            input.append(flow)
            input_array = np.array([np.asarray(input)])
            prediction = model.model.predict(input_array)[0][0]

    else:
        raise Exception('Sorry, the model type is not recognized')

    if prediction <= 10:
        speed_label = myFont.render('Slow', 1, blue)
    elif prediction > 10 and prediction <= 25:
        speed_label = myFont.render('Medium', 1, blue)
    elif prediction > 25 and prediction <= 40:
        speed_label = myFont.render('Fast', 1, blue)
    else:
        speed_label = myFont.render('Very Fast', 1, blue)

    A.append(prediction)
    line1.set_ydata(A)
    line1.set_xdata(range(len(A)))
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

    diceDisplay = myFont.render(str(prediction), 1, blue)
    screen.blit(randNumLabel, (50, 420))
    screen.blit(speed_label, (200, 420))
    screen.blit(diceDisplay, (50, 450))
    clock.tick(60)
    pygame.display.flip()


def validation_loop():

    # loading models
    # -------------------------------------------------
    model = i3d(weights_path='id3_64_8.h5', input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, configs.CHANNELS))

    # -------------------------------------------------
    # steerings and images
    steering_labels = path.join(configs.VAL_DIR, 'labels.csv')

    # read the steering labels and image path
    df_truth = pd.read_csv(steering_labels, usecols=['frame_id', 'steering_angle'], index_col=None)

    # Create second screen with matplotlibs
    fig = pylab.figure(figsize=[6.4, 1.6], dpi=100)
    ax = fig.gca()
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    line1, = ax.plot([], [], 'b.-', label='Human')
    line2, = ax.plot([], [], 'r.-', label='Model')
    A = []
    B = []
    ax.legend(loc='upper left', fontsize=8)

    myFont = pygame.font.SysFont("monospace", 18)
    randNumLabel = myFont.render('Human Steer Angle:', 1, blue)
    randNumLabel2 = myFont.render('Model Steer Angle:', 1, red)
    speed_ms = 5

    input = []

    for i in range(configs.LENGTH):
        file = configs.VAL_DIR + "center/" + str(df_truth['frame_id'].loc[i]) + ".jpg"
        img = helper.load_image(file)
        input.append(img)

    # Run through all images
    for i in range(configs.LENGTH, len(df_truth)):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        p = configs.VAL_DIR + "center/" + str(df_truth['frame_id'].loc[i]) + ".jpg"
        img = helper.load_image(p)
        input.pop(0)
        input.append(img)
        input_array = np.array([img])
        prediction = model.model.predict(input_array)[0][0]
        actual_steers = df_truth['steering_angle'].loc[i]  # * 0.1 - 8 * 0.0174533  # 1 degree right correction

        A.append(actual_steers)
        B.append(prediction)
        line1.set_ydata(A)
        line1.set_xdata(range(len(A)))
        line2.set_ydata(B)
        line2.set_xdata(range(len(B)))
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

        diceDisplay = myFont.render(str(round(actual_steers * PI_RAD, 4)), 1, blue)
        diceDisplay2 = myFont.render(str(round(prediction * PI_RAD, 4)), 1, red)
        screen.blit(randNumLabel, (50, 420))
        screen.blit(randNumLabel2, (400, 420))
        screen.blit(diceDisplay, (50, 450))
        screen.blit(diceDisplay2, (400, 450))
        clock.tick(60)
        pygame.display.flip()


if __name__ == "__main__":

    # cruise_control_viz_loop()
    test_loop()