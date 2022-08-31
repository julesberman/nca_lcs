import numpy as np


def DavisSkodje(t, u, PARAMETERS=[10]):
    """
    Davis-Skodje 2D System
    """
    [gamma] = PARAMETERS
    assert gamma > 1, 'gamma must be greater than 1'

    x, y = u.T
    v = np.column_stack([-x, -gamma*y + ((gamma-1)*x+gamma*x**2)/(1+x)**2])
    return v


def Duffing1D(t, u, PARAMETERS=[1, 1]):
    """
    Returns vector field for the Duffing oscillator.
    Number of model parameters: 2 . PARAMETERS = [alpha, beta]
    Functional form: v = (y, alpha*x - beta*x**3), with u = (x, y)

    Parameters
    ----------
    t : float
        Time. (This vector field is independent of time.)

    u : ndarray, shape(n,)
        Points in phase space.

    PARAMETERS : list of floats, optional
        Vector field parameters [alpha, beta]. Default is [1, 1].

    Returns
    -------
    v : ndarray, shape(n,)
        Vector field at points u and time t..
    """
    x, y = u.T
    # Hamiltonian Model Parameter
    alpha, beta = PARAMETERS
    v = np.column_stack([y, alpha*x - beta*x**3])
    return v


def DoubleGyre(t, u, PARAMETERS=[0, 0.1, 2*np.pi/10, 0, 0, 1, 0.25]):
    """
    Returns 2D Double Gyre vector field at time t, for an array of points in phase space.
    Number of model parameters: 6 . PARAMETERS = [phase_shift, A, phi, psi, mu, s, epsilon]
    Functional form: 

    vx = -pi*A*sin(pi*f(t + phase_shift, x)/s)*cos(pi*y/s) - mu*x
    vy =  pi*A*cos(pi*f(t + phase_shift, x)/s)*sin(pi*y/s)*df(t + phase_shift,x)/dx - mu*y

    with

    f(t, x)    = epsilon*sin(phi*t + psi)*x**2 + (1 - 2*epsilon*sin(phi*t + psi))*x
    df/dx(t,x) = 2*epsilon*sin(phi*t + psi)*x + (1 - 2*epsilon*sin(phi*t + psi))
    u = (x, y)
    Parameters
    ----------
    t : float
        fixed time-point of vector field, for all points in phase space.
    u : array_like, shape(n,)
        points in phase space to determine vector field at time t.
    PARAMETERS : list of floats
        vector field parameters
    Returns
    -------
    v : array_like, shape(n,)
        vector field corresponding to points u, in phase space at time t
    """
    x, y = u.T
    # model parameter
    phase_shift, A, phi, psi, mu, s, epsilon = PARAMETERS

    time = t + phase_shift
    # vector field components

    def f(t, x): return epsilon*np.sin(phi*t + psi) * \
        x**2 + (1-2*epsilon*np.sin(phi*t + psi))*x

    def df_dx(t, x): return 2*epsilon*np.sin(phi*t + psi) * \
        x + (1-2*epsilon*np.sin(phi*t + psi))
    v_x = -np.pi*A*np.sin(np.pi*f(time, x)/s)*np.cos(np.pi*y/s) - mu*x
    v_y = np.pi*A*np.cos(np.pi*f(time, x)/s) * \
        np.sin(np.pi*y/s)*df_dx(time, x) - mu*y
    v = np.column_stack([v_x, v_y])
    return v
