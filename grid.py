def Duffing_grid(N):

    x_min, x_max = [-1.6, 1.6]
    y_min, y_max = [-1, 1]
    Nx, Ny = [N, N]

    grid_parameters = [(x_min, x_max, Nx), (y_min, y_max, Ny)]

    return grid_parameters


def DoubleGyre_grid(N):

    x_min, x_max = [0, 2]
    y_min, y_max = [0, 1]
    Nx, Ny = [N, N]

    grid_parameters = [(x_min, x_max, Nx), (y_min, y_max, Ny)]

    return grid_parameters


def HamSaddle_grid(N):

    x_min, x_max = [-1.6, 1.6]
    y_min, y_max = [-1, 1]
    Nx, Ny = [N, N]

    grid_parameters = [(x_min, x_max, Nx), (y_min, y_max, Ny)]

    return grid_parameters


def DavisSkodje_grid(N):

    x_min, x_max = [0, 2]
    y_min, y_max = [0, 1.4]
    Nx, Ny = [N, N]

    grid_parameters = [(x_min, x_max, Nx), (y_min, y_max, Ny)]

    return grid_parameters
