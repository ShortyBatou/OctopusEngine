import numpy as np
import matplotlib.pyplot as plt

performance = [
    [0.4, 1.0, 1.3, 2.1, 2.4, 3.0, 3.3, 3.6, 3.8, 4.3, 4.5, 4.8, 6.0, 6.9, 7.9, 8.6, 11.2],  # P1
    [1.6, 4.2, 10.0],  # P2
    [0.6, 1.2, 1.7, 2.5, 2.8, 3.0, 3.9, 4.1, 4.3, 4.5, 4.8],  # Q1
    [7.5, 10.0, 12.0]  # Q2
]

size = [
    [0.4, 1.0, 1.3, 2.1, 2.4, 3.0, 3.3, 3.6, 3.8, 4.3, 4.5, 4.8, 6.0, 6.9, 7.9, 8.6, 11.2],  # P1
    [1.6, 4.2, 10.0],  # P2
    [0.6, 1.2, 1.7, 2.5, 2.8, 3.0, 3.9, 4.1, 4.3, 4.5, 4.8],  # Q1
    [7.5, 10.0, 12.0]  # Q2
]

error = [
    [7.50e-01, 1.50e-01, 4.20e-02, 1.50e-02, 7.10e-03, 3.30e-03, 1.70e-03, 1.00e-03, 7.50e-04, 6.10e-04, 5.40e-04, 3.90e-04, 2.60e-04, 1.40e-04, 1.10e-04, 7.89e-05, 4.96e-05],  # P1
    [3.50e-03, 1.06e-04, 3.80e-05],  # P2
    [1.60e-01, 2.40e-02, 6.00e-03, 2.10e-03, 8.70e-04, 4.30e-04, 2.70e-04, 1.60e-04, 6.40e-05, 5.30e-05, 2.25e-05],  # Q1
    [2.90e-03, 8.50e-05, 1.60e-05]  # Q2
]

fig, axes = plt.subplots(1, 1, figsize=(6, 5))

def display(axe, performance, error):
    name = ["P1", "P2", "Q1", "Q2"]
    colors = ["blue", "purple", "green", "cyan"]
    for i, (perf, err) in enumerate(zip(performance, error)):
        x = np.array(perf)
        y = np.array(err)

        # Régression polynomiale dans l’espace log-log
        logx, logy = np.log10(x), np.log10(y)
        coeffs = np.polyfit(logx, logy, deg=2)  # polynôme de degré 2ee<ee

        # Affichage
        axe.scatter(x, y, label=name[i], color = colors[i])

display(axes, performance, error)

titles = [""]
i = 0

axes.grid(False)
axes.set_yscale("log")
if(i == 0):
    axes.set_ylabel("MSE")
axes.set_xlabel("t (ms)")
axes.legend()
axes.set_title(titles[i])
i = i + 1
    
# Echelles log
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("01_plot.png")
plt.show()
