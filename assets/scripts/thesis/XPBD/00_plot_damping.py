import numpy as np
import matplotlib.pyplot as plt

damping = [
   [1.05e-05, 9.50e-06, 8.50e-06, 7.50e-06, 6.50e-06, 5.50e-06, 4.50e-06, 3.50e-06, 2.50e-06, 1.50e-06]
]

diff = [
    [2.10e-09, 6.30e-10, 1.30e-10, 2.90e-11, 3.90e-12, 3.60e-12, 3.60e-12, 3.60e-12, 3.60e-12, 4.00e-12]
]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

def display(axe, performance, error):
    colors = ["#1D1818"]
    for i, (perf, err) in enumerate(zip(performance, error)):
        x = np.array(perf)
        y = np.array(err)

        # Affichage
        axe.plot(x, y, linewidth=2, color = colors[i], linestyle="-")

display(ax, damping, diff)

titles = ["Impacte de l'atténuation sur la solution"]

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("MSE", fontsize=14) # Augmenté de 2 points
ax.set_xlim(1.5e-06, 1.05e-05)
ax.set_ylim(1.0e-12, 5e-09)
ax.set_xlabel("coefficient d'atténuation", fontsize=14) # Augmenté de 2 points

# Augmenter la taille des labels des ticks
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

x_val = 6.5e-6
ax.axvline(x=x_val, color='r', linestyle='--', linewidth=1)
ax.text(x_val, (ax.get_ylim()[0] * ax.get_ylim()[1])**0.5, f'{x_val:.2e}',
        color='r', ha='right', va='center', rotation=90, fontsize=14) # Augmenté de 2 points

ax.set_title(titles[0], fontsize=14) # Augmenté de 2 points (titre est souvent plus grand que les labels)

# Echelles log
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("00_plot_damping.png")
plt.show()