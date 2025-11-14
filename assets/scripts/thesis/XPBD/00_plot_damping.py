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
ax.set_ylabel("MSE")
ax.set_xlim(1.5e-06, 1.05e-05)
ax.set_ylim(1.0e-12, 5e-09)
ax.set_xlabel(r"k_d")

x_val = 6.5e-6
ax.axvline(x=x_val, color='r', linestyle='--', linewidth=1)
ax.text(x_val, (ax.get_ylim()[0] * ax.get_ylim()[1])**0.5, f'{x_val:.2e}',
        color='r', ha='right', va='center', rotation=90)

ax.set_title(titles[0])
    
# Echelles log
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("00_plot_damping.png")
plt.show()
