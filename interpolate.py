from scipy.interpolate import CubicSpline
import numpy as np


def find_lookback(future):
    x = [0, 15, 30]
    y = [5, 50, 100]

    # use bc_type = 'natural' adds the constraints as we described above
    f = CubicSpline(x, y, bc_type='natural')
    x_new = future
    y_new = int(f(x_new))
    print(y_new)

    return y_new