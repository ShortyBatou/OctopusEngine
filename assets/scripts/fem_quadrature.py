import numpy as np
from numpy.polynomial.legendre import leggauss

import quadpy


def f(x):
    return np.sin(x[0]) * np.sin(x[1])


triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.7, 0.5]])

# get a "good" scheme of degree 10
scheme = quadpy.t2.get_good_scheme(10)
val = scheme.integrate(f, triangle)