import numpy as np


def get_point_array_from_grid(grid_parameters):
    [(x_min, x_max, Nx), (y_min, y_max, Ny)] = grid_parameters
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    xv, yv = np.meshgrid(x, y)
    x, y = xv.flatten(), yv.flatten()
    pts = np.array([x, y]).T

    return pts
