import sys
import numpy as np
import matplotlib.pyplot as plt
h = [
    [0.707, 0.354, 0.283, 0.236, 0.177, 0.118, 0.101, 0.0884],
    [1.41, 0.707, 0.354],
    [1.73, 0.866, 0.433, 0.346, 0.289, 0.217, 0.144],
    [1.73, 0.866]
]


error = [
    [1.43e-1, 1.45e-2, 6.50e-3, 3.50e-3, 1.60e-3, 5.20e-4, 3.78e-4, 6.26e-4],
    [3.60e-3, 3.30e-4, 5.67e-5],
    [1.60e-1, 2.08e-2, 1.10e-3, 6.20e-4, 3.60e-4, 2.00e-4, 3.10e-5],
    [2.30e-3, 2.58e-5]
]

elem = ["P1", "P2", "Q1", "Q2" ]
colors = ["C0", "C1", "C2", "C3"]
order = [2,3,3,4]
slope = [1,1,1,1]

# Paramètres
L = 0

# Préparation de la figure
plt.figure(figsize=(8,6))
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.08, 1.8)
plt.ylim(1e-5, 1e-1, )
x = np.logspace(-2, 1, 200)

for i in range(len(h)):
    plt.plot(h[i], error[i], label=elem[i], linewidth=2, color=colors[i])
    plt.plot(x, slope[i]*x**(order[i]+3), '--', color=colors[i], label=r"$O(H^{%d})$" % (L+order[i]))

# Légendes et axes
plt.xlabel("h")
plt.ylabel("error")
plt.legend(loc='lower right', fontsize=12)

# Sortie PDF
plt.tight_layout()
plt.savefig(f"profiles.pdf")
plt.close()