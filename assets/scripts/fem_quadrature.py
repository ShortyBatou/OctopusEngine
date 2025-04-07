import numpy as np
from numpy.polynomial.legendre import leggauss

# n = nombre de points de quadrature
n = 2
x, w = leggauss(n)

print("Coordonnées (noeuds):", x)
print("Poids:", w)