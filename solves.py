import matplotlib.pyplot as plt
import numpy as np
from ldds.vector_fields import Duffing1D
from scipy.integrate import solve_ivp
import scipy
import scipy.signal
from einops import rearrange


def get_point_array_from_grid(grid):
    [(x_min, x_max, Nx), (y_min, y_max, Ny)] = grid
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    xv, yv = np.meshgrid(x, y)
    x, y = xv.flatten(), yv.flatten()
    pts = np.array([x, y]).T

    return pts


def integrate_ivp(vector_field, grid, t_n, return_sol=False, t_0=0, rtol=1e-5, atol=1e-12, max_step=np.inf):

    ax1 = np.linspace(*grid[0])
    ax2 = np.linspace(*grid[1])

    Ax1, Ax2 = np.meshgrid(ax1, ax2)
    dims_slice_axes = np.column_stack((Ax1.ravel(), Ax2.ravel()))
    y0 = dims_slice_axes.ravel()
    num_traj = dims_slice_axes.shape[0]

    def f(t, y):
        u = y.reshape((-1, 2))
        return vector_field(t, u).ravel()

    solution = solve_ivp(f, [t_0, t_n], y0, rtol=rtol,
                         atol=atol, max_step=max_step)

    trajs = solution.y.reshape(num_traj, 2, -1)

    res = trajs
    if return_sol:
        res = trajs, solution

    return res


def solve_FTLE(trajs, grid_parameters, tau):

    [(x_min, x_max, Nx), (y_min, y_max, Ny)] = grid_parameters
    central_diff = np.array([[1, 0, -1]])

    trajs_end = trajs[..., -1]
    trajs_grid = trajs_end.reshape(Nx, Ny, 2)
    tX, tY = trajs_grid[..., 0], trajs_grid[..., 1]
    dX = scipy.signal.convolve2d(tX, central_diff, 'same')
    dY = scipy.signal.convolve2d(tY, central_diff.T, 'same')
    del_x, del_y = (x_max - x_min) / (Nx-1), (y_max - y_min) / (Ny-1)

    jac = np.array([[dX / del_x*2, dX / del_y*2],
                   [dY / del_x*2, dY / del_y*2]])

    jac = rearrange(jac, 'j1 j2 nx ny -> (nx ny) j1 j2')

    N = jac.shape[0]
    FTLE = np.zeros(N)
    FTLV = np.zeros((N, 2))
    for i, j in enumerate(jac):
        U, s, Vh = scipy.linalg.svd(j)
        FTLE[i] = s[0]
        FTLV[i] = Vh[:, 0] * np.sign(Vh[-1, 0])

    FTLE = np.log(FTLE) * (1/np.abs(tau))
    # FTLE = FTLE.reshape(Nx, Ny)
    # FTLV = FTLV.reshape(Nx, Ny, 2)

    return FTLE, FTLV


def mean_tse(trajs, dt):
    central_diff = np.array([[1, 0, -1]])

    trajs_x = trajs[:, 0, :]
    trajs_y = trajs[:, 1, :]
    dtx = scipy.signal.convolve2d(trajs_x, central_diff, 'valid') / (2*dt)
    dty = scipy.signal.convolve2d(trajs_y, central_diff, 'valid') / (2*dt)

    eps = 1e-12
    dtx += eps
    dty += eps
    xtse = np.sum(
        np.abs(np.log(np.abs(dtx[:, 1:])/np.abs(dtx[:, :-1])+eps)), axis=1)
    ytse = np.sum(
        np.abs(np.log(np.abs(dty[:, 1:])/np.abs(dty[:, :-1])+eps)), axis=1)

    tse = np.linalg.norm(np.vstack((xtse, ytse)), axis=0)

    return tse


def learn_lcs(X, Y, num_neurons, eta=0.1, epochs=1, eps=0):

    (N, dim) = X.shape
    neurons = []
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    for i in range(num_neurons):
        rand_x = np.array([np.random.uniform(low=x_min, high=x_max),
                           np.random.uniform(low=y_min, high=y_max)])
        neurons.append((rand_x, np.inf))

    def update_neurons(neurons, cur):
        (xt, yt) = cur
        opt_dist = np.inf
        opt_i = -1
        for i, (xj, yj) in enumerate(neurons):
            dist = np.linalg.norm(xt-xj)
            if yt < yj + eps and dist < opt_dist:
                opt_dist = dist
                opt_i = i

        if opt_i != -1:
            x, y = neurons[opt_i]
            if y == np.inf:
                y = yt
            else:
                y += eta*(yt-y)
            x += eta*(xt - x)

            # # enforce min dist
            # no_overlap = True
            # for i, (xj, yj) in enumerate(neurons):
            #     d = np.linalg.norm(x-xj)
            #     if d < min_dist and i != opt_i:
            #         no_overlap = False

            # if no_overlap:
            neurons[opt_i] = (x, y)

        return neurons

    for epoch in range(epochs):
        shuffle_i = np.arange(N)
        np.random.shuffle(shuffle_i)
        X, Y = X[shuffle_i], Y[shuffle_i]
        for x, y in zip(X, Y):
            neurons = update_neurons(neurons, (x, y))

    return neurons
