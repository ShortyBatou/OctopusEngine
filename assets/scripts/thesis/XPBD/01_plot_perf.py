import numpy as np
import matplotlib.pyplot as plt

p1_performance = [
    [2.6, 3.75, 5.99, 7.48, 9.13, 11, 12, 15.9, 16.7, 19.8, 22.8, 25.3, 29.1, 33.1, 37.1, 39.3, 43.8, 52.8, 65.7, 73.3], # Q1
    [1.86e+00, 4.05e+00, 5.63e+00, 7.27e+00, 9.49e+00, 1.11e+01, 1.31e+01, 1.48e+01, 1.64e+01, 1.88e+01, 2.37e+01, 2.67e+01, 3.04e+01, 3.39e+01, 3.76e+01, 4.33e+01, 5.03e+01, 5.80e+01, 6.51e+01, 71.9] # Q2
]

p1_error = [
    [6.15,2.12,0.94,0.463,0.255,0.155,0.0993,0.06846,0.0492,0.0365,0.0232,0.0152,0.0103,0.00821,0.0066,0.00354,0.00226,0.0035,0.0009,0.00089],# Q1
    [2.52e+00, 3.20e-01, 7.01e-02, 2.27e-02, 9.27e-03, 4.51e-03, 2.38e-03, 1.44e-03, 5.66e-04, 8.98e-04, 2.95e-04, 1.48e-04, 7.89e-05, 8.74e-05, 1.32e-04, 1.45e-05, 2.60e-05, 2.73e-04, 4.24e-05, 1.00e-04] # Q2
]

# --- Données ---
p2_performance = [
    [3.97e+00, 8.00e+00, 1.18e+01, 1.58e+01, 1.95e+01, 2.34e+01, 2.75e+01, 3.12e+01, 3.56e+01, 3.91e+01, 4.85e+01, 5.74e+01, 6.43e+01, 7.39e+01, 7.96e+01, 9.43e+01, 1.11e+02, 1.26e+02, 1.44e+02, 1.58e+02], # Q1
    [3.99e+00, 8.09e+00, 1.19e+01, 1.60e+01, 1.99e+01, 2.33e+01, 2.76e+01, 3.13e+01, 3.57e+01, 3.91e+01, 4.79e+01, 5.67e+01, 6.43e+01, 7.34e+01, 8.05e+01, 9.69e+01, 1.13e+02, 1.29e+02, 1.47e+02, 1.61e+02] # Q2
]

p2_error = [
    [1.02e+01, 6.87e+00, 3.80e+00, 2.55e+00, 1.75e+00, 1.21e+00, 8.83e-01, 6.31e-01, 4.74e-01, 3.56e-01, 2.26e-01, 1.46e-01, 9.76e-02, 7.29e-02, 5.79e-02, 3.51e-02, 2.50e-02, 2.17e-02, 1.45e-02, 8.44e-03],# Q1
    [8.00e+00, 2.30e+00, 7.33e-01, 2.38e-01, 1.60e-01, 5.77e-02, 3.13e-02, 1.83e-02, 1.14e-02, 7.43e-03, 3.47e-03, 1.77e-03, 9.70e-04, 7.86e-04, 5.66e-04, 8.79e-05, 1.00e-04, 5.90e-04, 1.39e-05, 3.41e-05] # Q2
]


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes = axes.flatten()

def display(axe, performance, error):
    name = ["SHN", "ours"]
    colors = ["#3498db", "#8e44ad"]
    for i, (perf, err) in enumerate(zip(performance, error)):
        x = np.array(perf)
        y = np.array(err)

        # Régression polynomiale dans l’espace log-log
        logx, logy = np.log10(x), np.log10(y)
        coeffs = np.polyfit(logx, logy, deg=2)  # polynôme de degré 2ee<ee
        poly = np.poly1d(coeffs)

        # Courbe lissée
        x_smooth = np.linspace(x.min(), x.max(), 300)
        y_smooth = 10**poly(np.log10(x_smooth))

        # Affichage
        axe.scatter(x, y, label=name[i], color = colors[i])
        axe.plot(x_smooth, y_smooth, linewidth=2, color = colors[i], linestyle="-")

display(axes[0], p1_performance, p1_error)
display(axes[1], p2_performance, p2_error)

titles = ["P1", "P2"]
i = 0
for ax in axes:
    ax.grid(False)
    ax.set_yscale("log")
    if(i == 0):
        ax.set_ylabel("MSE")
    ax.set_xlabel("t (ms)")
    ax.legend()
    ax.set_title(titles[i])
    i = i + 1
    
# Echelles log
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("01_plot.png")
plt.show()
