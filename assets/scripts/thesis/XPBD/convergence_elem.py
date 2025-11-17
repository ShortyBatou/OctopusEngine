import sys
import numpy as np
import matplotlib.pyplot as plt
h = [
    [7.07e-1, 3.54e-1,2.83e-1,2.36e-1,1.77e-1, 1.18e-1, 1.01e-1, 8.84e-2],
    [1.41,7.07e-1,3.54e-1],
    [1.73,8.66e-1,4.33e-1,3.46e-1,2.89e-1,2.17e-1,1.44e-1],
    [1.73,8.66e-1]
]


h_mean = [
    [7.07e-1, 3.54e-1,2.83e-1,2.36e-1,1.77e-1, 1.18e-1, 1.01e-1, 8.84e-2],
    [1.41,7.07e-1,3.54e-1],
    [1.31,0.677,0.346,0.278,0.232,0.175,0.117],
    [1.31,0.677]
]


nb_elem = [
    [160,1280,2500,4320,10240,34560,54880,81920],
    [20,160,540],
    [4,32,256,500,864,2048,6912],
    [4,32]
]

for i in range(len(h)): 
    for j in range(len(nb_elem[i])):
        nb_elem[i][j] = (4. / nb_elem[i][j])

error = [
    [1.43e-1, 1.45e-2, 6.50e-3, 3.50e-3, 1.60e-3, 5.20e-4, 3.78e-4, 6.26e-4],
    [3.60e-3, 3.30e-4, 5.67e-5],
    [1.60e-1, 2.08e-2, 1.10e-3, 6.20e-4, 3.60e-4, 2.00e-4, 3.10e-5],
    [2.30e-3, 2.58e-5]
]

colors = ["#3498db", "#8e44ad", "#62c467", "#16a085"]
elem = ["P1", "P2", "Q1", "Q2" ]
order = [2,3,3,4]
slope = [0.1,0.001,0.015,0.00008]

# Paramètres
L = 0

# Préparation de la figure
plt.figure(figsize=(8,6))
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-5, 1e-1, )
x = np.logspace(-2, 1, 200)

for i in range(len(h)):
    plt.plot(h_mean[i], error[i], label=elem[i], linewidth=2, color=colors[i])
    plt.plot(x, slope[i]*x**(order[i]), '--', color=colors[i], label=r"$O(H^{%d})$" % (L+order[i]))

# Légendes et axes
plt.xlabel("h")
plt.ylabel("error")
plt.legend(loc='upper left', fontsize=12)

# Sortie PDF
plt.tight_layout()
plt.savefig(f"profiles.pdf")
plt.close()