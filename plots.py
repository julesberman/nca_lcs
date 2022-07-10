
import matplotlib.pyplot as plt
import numpy as np
from tools import *


def plot_vector_field(vector_field, grid,  size=(14, 8), T=0):

    [(x_min, x_max, Nx), (y_min, y_max, Ny)] = grid
    pts = get_point_array_from_grid(grid)
    [x, y] = pts.T
    v_pts = vector_field(T, pts)
    u, v = v_pts[:, 0], v_pts[:, 1]

    plt.quiver(x, y, u, v)
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    fig = plt.gcf()
    fig.set_size_inches(*size)
    plt.show()


# def plot_vector_field(grid, v_pts, size=(14, 8), N=25, T=0):

#     [(x_min, x_max, Nx), (y_min, y_max, Ny)] = grid

#     pts = get_point_array_from_grid([(x_min, x_max, N), (y_min, y_max, N)])
#     [x, y] = pts.T

#     d = v_pts.shape[0] // pts.shape[0]
#     v_pts = v_pts[::d]

#     u, v = v_pts[:, 0], v_pts[:, 1]
#     print(x.shape, u.shape)
#     plt.quiver(x, y, u, v)
#     plt.xlim([x_min, x_max])
#     plt.ylim([y_min, y_max])
#     fig = plt.gcf()
#     fig.set_size_inches(*size)
#     plt.show()


def plot_scalar_field(grid, z, size=(14, 8), show=True):

    ax1 = np.linspace(*grid[0])
    ax2 = np.linspace(*grid[1])
    z = z.reshape(grid[0][-1], grid[1][-1])
    plt.contourf(ax1, ax2, z, cmap='viridis', levels=100)
    fig = plt.gcf()
    plt.colorbar()

    fig.set_size_inches(*size)

    if show:
        plt.show()

    else:
        return fig
