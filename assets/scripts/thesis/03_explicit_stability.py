import numpy as np
import matplotlib.pyplot as plt

# Paramètres du système
omega = 10  # fréquence
x0 = 1.0    # position initiale
v0 = 0.0    # vitesse initiale
t_max = 2.0

# Solution exacte : x(t) = cos(omega * t)
dt_exact = 0.001
t_exact = np.arange(0, t_max + dt_exact, dt_exact)
x_exact = np.cos(omega * t_exact)

# Pas de temps à tester
dt_values = [0.0001, 0.005, 0.01]
colors = ['red', 'green', 'blue']
titles = ["dt = 1e-4", "dt = 5e-3", "dt = 1e-2"]

# Création d'une grille 2x3 : haut = x(t), bas = erreur
fig, axes = plt.subplots(2, 3, figsize=(16, 6), sharex='col',  sharey='row')

for col, (dt, title, color) in enumerate(zip(dt_values, titles, colors)):
    t_values = np.arange(0, t_max + dt, dt)
    x_values = [x0]
    v_values = [v0]

    for t in t_values[:-1]:
        x_curr = x_values[-1]
        v_curr = v_values[-1]

        # Euler explicite
        x_next = x_curr + dt * v_curr
        v_next = v_curr - dt * omega**2 * x_curr

        x_values.append(x_next)
        v_values.append(v_next)

    # Interpolation de la solution exacte sur les t_values
    x_exact_interp = np.cos(omega * t_values)
    error = np.abs(np.array(x_values) - x_exact_interp)

    # Graphique supérieur : solution
    axes[0, col].plot(t_exact, x_exact, '--', color='black')
    axes[0, col].plot(t_values, x_values, '-', color=color)
    axes[0, col].axis([0, t_max, -2,2])
    axes[0, col].set_title(title, fontsize=18)
    if(col == 0):
        axes[0, col].set_ylabel("x(t)", fontsize=16)
    axes[0, col].grid(False)

    # Graphique inférieur : erreur
    axes[1, col].plot(t_values, error, '-', color='darkred')
    axes[1, col].axis([0, t_max, 0,1])
    axes[1, col].set_xlabel("t", fontsize=16)
    if(col == 0):
        axes[1, col].set_ylabel("Erreur Intégration", fontsize=16)
    axes[1, col].grid(False)

# Mise en page
plt.tight_layout()
plt.savefig('explicit_stability.png', dpi=200)
plt.show()
