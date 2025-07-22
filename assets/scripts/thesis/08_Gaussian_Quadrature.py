import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from scipy.integrate import quad

# Points et poids de Gauss à 2 points sur [-1, 1]
gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
gauss_weights = [1, 1]

# Génère les polynômes de degré 1 à 4
degrees = [1, 2, 3, 4]

x = np.linspace(-1, 1, 400)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs = axs.ravel()

labels = ["$f(x) = 2x + 1$", "$f(x) = 3x^2 + 2x + 1$", "$f(x) = 4x^3 + 3x^2 + 2x + 1$", "$f(x) = 5x^4 + 4x^3 + 3x^2 + 2x + 1$"]

for i, deg in enumerate(degrees):
    # Crée un polynôme aléatoire de degré `deg` avec des coefficients simples
    coeffs = np.arange(1, deg + 2)  # Par exemple : [1, 2] pour deg=1
    p = Polynomial(coeffs)
    y = p(x)

    # Valeur exacte de l'intégrale
    exact_integral, _ = quad(p, -1, 1)

    # Valeur approchée par quadrature de Gauss
    approx_integral = sum(w * p(xi) for w, xi in zip(gauss_weights, gauss_points))
    samples = [p(xi) for xi in gauss_points]
    # Tracé
    axs[i].plot(x, y, label= labels[i])
    axs[i].fill_between(x, y, alpha=0.1, color='grey')
    axs[i].vlines(gauss_points, 0, samples, color='red', linestyle='--')
    axs[i].scatter(gauss_points, samples, alpha=1, color='red', label='Quadratures')
    axs[i].set_xlim([-1, 1])
    
    
    axs[i].set_title(f"$\int f(x) dx$ ≈ {approx_integral:.4f} | Erreur = {abs(approx_integral - exact_integral):.4f}")
    axs[i].axhline(0, color='black', linewidth=0.5)
    axs[i].axvline(0, color='black', linewidth=0.5)
    axs[i].legend()
    axs[i].grid(False)

plt.suptitle("Intégration de Gauss à 2 points", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("08_gaussian_quadrature.png")
plt.show()

