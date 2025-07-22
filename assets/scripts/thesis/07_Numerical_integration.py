import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import roots_legendre

# Fonction à intégrer
def f(x):
    return np.sin(x**2)

# Intervalle d'intégration
a, b = 0, np.sqrt(np.pi)

# Intégrale analytique
I_exact, _ = quad(f, a, b)

# Préparer figure
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes = axes.flatten()

# Paramètres
N = 10  # nombre de points pour méthodes discrètes
x_dense = np.linspace(a, b, 1000)
y_dense = f(x_dense)

# 1. Trapèze
x_trap = np.linspace(a, b, N)
y_trap = f(x_trap)
I_trap = np.trapz(y_trap, x_trap)

axes[0].plot(x_dense, y_dense, 'k', label='f(x)')
axes[0].fill_between(x_trap, y_trap, alpha=0.3)
axes[0].plot(x_trap, y_trap, 'o-', color='blue')
axes[0].set_title("Trapèzes")
axes[0].text(0.5, -0.4,
             f"$\int f(x) dx$ ≈ {I_trap:.5f} | Erreur = {abs(I_trap - I_exact):.2e}",
             ha='center', transform=axes[0].transAxes)

# 2. Monte Carlo
np.random.seed(1)
x_rand = np.random.uniform(a, b, N)
y_rand = f(x_rand)
I_montecarlo = (b - a) * np.mean(y_rand)

axes[1].plot(x_dense, y_dense, 'k', label='f(x)')
axes[1].scatter(x_rand, y_rand, alpha=1, color='green', label='Samples')
axes[1].vlines(x_rand, 0, f(x_rand), color='green', linestyle='--')
axes[1].set_title("Monte Carlo")
axes[1].text(0.5, -0.4,
             f"$\int f(x) dx$ ≈ {I_montecarlo:.5f} | Erreur = {abs(I_montecarlo - I_exact):.2e}",
             ha='center', transform=axes[1].transAxes)

# 3. Quadrature de Gauss (Gauss-Legendre)
# On mappe de [-1, 1] vers [a, b]
nodes, weights = roots_legendre(N)
# Transformation des points
x_gauss = 0.5 * (nodes + 1) * (b - a) + a
w_gauss = 0.5 * (b - a) * weights
I_gauss = np.sum(w_gauss * f(x_gauss))

axes[2].plot(x_dense, y_dense, 'k', label='f(x)')
axes[2].vlines(x_gauss, 0, f(x_gauss), color='red', linestyle='--')
axes[2].plot(x_gauss, f(x_gauss), 'o', color='red', label='Points de Gauss')
axes[2].set_title("Quadrature de Gauss")
axes[2].text(0.5, -0.4,
             f"$\int f(x) dx$ ≈ {I_gauss:.5f} | Erreur = {abs(I_gauss - I_exact):.2e}",
             ha='center', transform=axes[2].transAxes)


# Mise en forme
for ax in axes:
    ax.set_xlim(a, b)
    ax.set_ylim(0, 1.1)
    ax.grid(False)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

plt.suptitle("Méthodes d'intégration numérique", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("07_numerical_integration.png")
plt.show()
