# taken from the git repository 
# https://github.com/jasonweiyi/understanding_gaussian_process/blob/master/generate_training_data.py
# from the medium article "understanding GPs"
import numpy as np
import matplotlib.pyplot as plt


def generate_points(start, end, section=7, quota=[5, 10], noise=0.1):
    """
    Generate data points from start to end. The data points are generated
    in a number of sections. Odd sections contain the number of points
    given by quota[0]. Even sections contain the number of points given
    by quota[1]. 0-mean Gaussian with variance noise is added into each data point.
    """
    np.random.seed(1234)
    x = []

    section_length = (end - start) / float(section)
    for s in range(section):
        section_start = s * section_length + start
        section_end = (s + 1) * section_length + start

        if s % 2 == 0:
            points = quota[0]
        else:
            points = quota[1]
        x.extend(np.linspace(section_start, section_end, points).tolist())

    x = np.array(x)
    y = x*np.sin(x) + np.random.normal(scale=noise, size=len(x))

    return x, y
