import numpy as np
import matplotlib.pyplot as plt

# Définir la fonction à discrétiser
def f(t):
    A = 0.5      # amplitude
    mu = 10.0     # moyenne
    sigma = 4.0  # écart-type
    return A * np.exp(-((t - mu)**2) / (2 * sigma**2))
# Paramètres de discrétisation
t_min = 0
t_max = 20
dt = 2.5

# Vecteur de temps discrétisé
t_values = np.arange(t_min, t_max + 0.01, 0.01)
f_values = f(t_values)

t_points = np.arange(t_min, t_max + dt, dt)
f_points = f(t_points)

plt.figure(figsize=(18, 6))  # Largeur:Hauteur = 2:1


# Coloration des zones sous la courbe avec intensité croissante
n = len(t_values) - 1
m = len(t_points) - 1
for i in range(n):
    t0, t1 = t_values[i], t_values[i+1]
    f0, f1 = f_values[i], f_values[i+1]
    
    # Zone à remplir (trapèze ou rectangle)
    t_fill = [t0, t1]
    f_fill = [f0, f1]

    # Calcul d'une intensité de rouge croissante
    progression = i / n
    red_intensity = int(progression * m) / m  # va de 0.1 à 1.0
    
    plt.fill_between(t_fill, [0, 0], f_fill, color=(0.8-red_intensity*0.7, 0.8-red_intensity*0.7, 1 - red_intensity * 0.2), alpha=0.5)

for i, (t, f) in enumerate(zip(t_points, f_points)):
    label = "0" if i == 0 else f"{i}h"
    plt.text(t, f + 0.025, label, ha='center', va='bottom', fontsize=18, color='black')

# Tracé de la courbe
plt.plot(t_values, f_values, color='black', label='')

# Traits verticaux
plt.vlines(t_points, 0, f_points, colors=(0,0,0), linestyles='dashed', linewidth=1)
plt.scatter(t_points, f_points, color=(0,0,0), label='')
axes = plt.gca()
axes.set_ylim([0,0.6])


# Mise en forme
plt.xlabel('t', fontsize=18)
plt.ylabel('f(t)', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(False)
plt.savefig('discretize_function.png', dpi=200)
plt.show()
