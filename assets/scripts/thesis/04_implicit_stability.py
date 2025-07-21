import numpy as np
import matplotlib.pyplot as plt

# Paramètres du système
omega = 10
x0 = 1.0
v0 = 0.0
t_max = 2.0

# Solution exacte
dt_exact = 0.001
t_exact = np.arange(0, t_max + dt_exact, dt_exact)
x_exact = np.cos(omega * t_exact)

# Pas de temps testés
dt_values = [0.0001, 0.005, 0.05]
colors = ['red', 'green', 'blue']
titles = ["dt = 1e-4", "dt = 5e-3", "dt = 1e-2"]

# Grille de graphes
fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex='col')

for col, (dt, title, color) in enumerate(zip(dt_values, titles, colors)):
    t_values = np.arange(0, t_max + dt, dt)
    x_values = [x0]
    v_values = [v0]

    # Constantes pour le schéma implicite
    denom = 1 + (dt**2) * omega**2
    A11 = 1 / denom
    A12 = dt / denom
    A21 = -dt * omega**2 / denom
    A22 = 1 / denom

    # Intégration implicite
    for _ in t_values[:-1]:
        x_n = x_values[-1]
        v_n = v_values[-1]

        x_next = A11 * x_n + A12 * v_n
        v_next = A21 * x_n + A22 * v_n

        x_values.append(x_next)
        v_values.append(v_next)

    # Interpolation exacte
    x_exact_interp = np.cos(omega * t_values)
    error = np.abs(np.array(x_values) - x_exact_interp)

    # Tracé solution
    axes[0, col].plot(t_exact, x_exact, '--', color='black')
    axes[0, col].plot(t_values, x_values, '-', color=color)
    axes[0, col].set_title(title)
    axes[0, col].set_ylabel("x(t)")
    axes[0, col].legend()
    axes[0, col].grid(False)

    # Tracé erreur
    axes[1, col].plot(t_values, error, '-', color='darkred')
    axes[1, col].set_xlabel("t")
    axes[1, col].axis([0, t_max, 0,1])
    axes[1, col].set_ylabel("Erreur Intégration")
    axes[1, col].grid(False)

# Mise en page
plt.suptitle("Stabilité d'Euler implicite", fontsize=18)
plt.savefig('04_implicit_stability.png', dpi=200)
plt.show()