import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_point_array_from_grid(grid_parameters):
    [(x_min, x_max, Nx), (y_min, y_max, Ny)] = grid_parameters
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    xv, yv = np.meshgrid(x, y)
    x, y = xv.flatten(), yv.flatten()
    pts = np.array([x, y]).T

    return pts


def make_plt_ani(plotter, data, **kwarg):

    fig = plt.figure()
    ims = []

    for d in data:
        plt.axes()
        plotter(d)
        ims.append(plt.gca().get_children())

    # default args, overwrite with user specified
    args = {'interval': 50, 'blit': True, 'repeat_delay': 500}
    for k, v in kwarg.items():
        args[k] = v

    ani = animation.ArtistAnimation(fig, ims, **args)

    plt.close()

    return ani


def make_plt_ani(plotter, data, **kwarg):

    fig = plt.figure()
    ims = []

    for d in data:
        plt.axes()
        plotter(d)
        ims.append(plt.gca().get_children())

    # default args, overwrite with user specified
    args = {'interval': 50, 'blit': True, 'repeat_delay': 500}
    for k, v in kwarg.items():
        args[k] = v

    ani = animation.ArtistAnimation(fig, ims, **args)

    plt.close()

    return ani
