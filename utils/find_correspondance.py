from pathlib import Path
import csv

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
colors = ['red', 'green', 'blue', 'yellow']


def onclick(event, points_left, points_right, ax):
    x, y = int(event.xdata), int(event.ydata)
    if event.inaxes == ax[0]:
        color = colors[len(points_left)]
        points_left.append((x, y, color))
        ax[0].scatter(x, y, c=color)
        plt.draw()
    elif event.inaxes == ax[1]:
        if len(points_left) == 0:
            return
        x_, y_, color = points_left[-1]
        points_right.append((x, y))
        points_left[-1] = (x_, y_)
        ax[1].scatter(x, y, c=color)
        plt.draw()
        save_correspondence_points(points_left, points_right, save_path)


def select_correspondence_points(left_image, right_image):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(left_image, cmap='gray')
    ax[0].set_title('Dynamic')
    ax[1].imshow(right_image, cmap='gray')
    ax[1].set_title('Static')
    points_left = []
    points_right = []
    fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, points_left, points_right, ax))
    plt.show()
    return points_left, points_right


def save_correspondence_points(points_left, points_right, save_path):
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Colors', 'DYN_X', 'DYN_Y', 'STATIC_X', 'STATIC_Y'])
        for i, ((x_left, y_left), (x_right, y_right)) in enumerate(zip(points_left, points_right)):
            writer.writerow([colors[i], x_left, y_left, x_right, y_right])


if __name__ == '__main__':
    idx = 0

    path = Path('rawData')
    save_path = path / f'points_{idx}.csv'

    left_img = np.load(path / f'left_{idx}.npy')
    right_img = np.load(path / f'right_{idx}.npy')

    points_left, points_right = select_correspondence_points(left_img, right_img)
    save_correspondence_points(points_left, points_right, save_path)
