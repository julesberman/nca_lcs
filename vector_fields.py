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
