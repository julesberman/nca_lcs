from tkinter import W
from turtle import pos
import numpy as np
from scipy.integrate import solve_ivp
import scipy
import scipy.signal
from einops import rearrange
import matplotlib.pyplot as plt
import copy


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


def mean_tse_wrong(trajs, dt):
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


def mean_tse(trajs, dt):
    central_diff = np.array([[1, 0, -1]])

    trajs_x = trajs[:, 0, :]
    trajs_y = trajs[:, 1, :]
    dtx = scipy.signal.convolve2d(trajs_x, central_diff, 'valid') / (2*dt)
    dty = scipy.signal.convolve2d(trajs_y, central_diff, 'valid') / (2*dt)

    norm = np.linalg.norm(np.array([dtx, dty]), axis=0)
    norm += 1e-12
    tse = np.sum(
        np.abs(np.log(norm[:, 1:]/norm[:, :-1])), axis=1)

    return tse


def learn_lcs(X, Y, num_neurons, eta=0.1, epochs=1, eps=0, min_dist=0):

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

            # enforce min dist
            no_overlap = True
            if min_dist > 0:
                for i, (xj, yj) in enumerate(neurons):
                    d = np.linalg.norm(x-xj)
                    if d < min_dist and i != opt_i:
                        no_overlap = False

            if no_overlap:
                neurons[opt_i] = (x, y)

        return neurons

    all_neurons = []
    for epoch in range(epochs):
        shuffle_i = np.arange(N)
        np.random.shuffle(shuffle_i)
        X, Y = X[shuffle_i], Y[shuffle_i]
        for x, y in zip(X, Y):
            neurons = update_neurons(neurons, (x, y))
            all_neurons.append(copy.deepcopy(neurons))

    return neurons, all_neurons


def min_field_cluster(X, Y, n_centroids, alpha=1.0, eta=0.1, epochs=1, return_history=False):
    '''
    Returns the sum of two decimal numbers in binary digits.
    P = number_of_sampled_points, D = dim_of_space, T = total_train_steps

            Parameters:
                    X (PxD matrix):        matrix indicating the positon of each points
                    Y (P vector):          indicates the slowness each point
                    alpha (float):         parameter to control weighting function
                    eta (float):           learning rate
                    epochs (int):          controls how many times we iterate through the entire training set (X, Y)
                    return_history (bool): weather to return a per step history of position and slowness
            Returns:
                    position (str): Binary string of the sum of a and b
                    slowness (str): Binary string of the sum of a and b
                    history (optina): Binary string of the sum of a and b
    '''

    def update_neurons(position, slowness, p_t, s_t):
        '''
        Helper function to perform one update step of the algorithm 
        N = n_centroids, D = dim_of_space

                Parameters:
                        position (PxD matrix): the positon of each centroid
                        slowness (N vector):   the current slowness of each centroid 
                        p_t (D vector):        the position of the incoming data point (trajectory start)
                        s_t (float):           the slowness of the incoming data point (trajectory slowness)
                Returns:
                        (position, slowness) updated as defined above

        '''

        # computer distance from all centroids to point
        dist = np.linalg.norm(position-p_t, axis=1)

        # find closest centroid position
        opt_i = np.argmin(dist)
        p_i = position[opt_i]
        s_i = slowness[opt_i]

        # compute update weight
        w = 1 if s_t < s_i else alpha/(s_t - s_i + alpha)

        # update
        position[opt_i] += eta*w*(p_t-p_i)
        slowness[opt_i] += eta*w*(s_t-s_i)

        return position, slowness

    # inititalize centroids by setting their position and slowness to
    # random samples from the training set
    inits_i = np.random.choice(X.shape[0], size=n_centroids)
    position = X[inits_i]
    slowness = Y[inits_i]

    all_positions = []
    all_slowness = []
    for _ in range(epochs):
        # for each epoch shuffle training set then learn
        shuffle_i = np.arange(X.shape[0])
        np.random.shuffle(shuffle_i)
        X, Y = X[shuffle_i], Y[shuffle_i]
        for x, y in zip(X, Y):
            position, slowness = update_neurons(position, slowness, x, y)

            # record history
            if return_history:
                all_slowness.append(slowness.copy())
                all_positions.append(position.copy())

    if return_history:
        return position, slowness, (all_positions, all_slowness)

    return position, slowness
