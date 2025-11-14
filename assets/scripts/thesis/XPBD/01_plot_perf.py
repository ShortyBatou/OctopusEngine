import numpy as np
import matplotlib.pyplot as plt

p1_performance = [
    [2.6, 3.75, 5.99, 7.48, 9.13, 11, 12, 15.9, 16.7, 19.8, 22.8, 25.3, 29.1, 33.1, 37.1, 39.3, 43.8, 52.8, 65.7, 73.3], # MM21
    [1.86e+00, 4.05e+00, 5.63e+00, 7.27e+00, 9.49e+00, 1.11e+01, 1.31e+01, 1.48e+01, 1.64e+01, 1.88e+01, 2.37e+01, 2.67e+01, 3.04e+01, 3.39e+01, 3.76e+01, 4.33e+01, 5.03e+01, 5.80e+01, 6.51e+01, 71.9] # Ours
]

p1_error = [
    [6.15,2.12,0.94,0.463,0.255,0.155,0.0993,0.06846,0.0492,0.0365,0.0232,0.0152,0.0103,0.00821,0.0066,0.00354,0.00226,0.0035,0.0009,0.00089],# MM21
    [2.52e+00, 3.20e-01, 7.01e-02, 2.27e-02, 9.27e-03, 4.51e-03, 2.38e-03, 1.44e-03, 8.98e-04, 5.66e-04, 2.95e-04, 1.48e-04, 7.89e-05, 8.74e-05, 1.32e-04, 1.45e-05, 2.60e-05, 2.73e-04, 4.24e-05, 1.00e-04] # Ours
]

# --- Données ---
p2_performance = [
    [3.97e+00, 8.00e+00, 1.18e+01, 1.58e+01, 1.95e+01, 2.34e+01, 2.75e+01, 3.12e+01, 3.56e+01, 3.91e+01, 4.85e+01, 5.74e+01, 6.43e+01, 7.39e+01, 7.96e+01, 9.43e+01, 1.11e+02, 1.26e+02, 1.44e+02, 1.58e+02], # MM21
    [3.99e+00, 8.09e+00, 1.19e+01, 1.60e+01, 1.99e+01, 2.33e+01, 2.76e+01, 3.13e+01, 3.57e+01, 3.91e+01, 4.79e+01, 5.67e+01, 6.43e+01, 7.34e+01, 8.05e+01, 9.69e+01, 1.13e+02, 1.29e+02, 1.47e+02, 1.61e+02] # Ours
]

p2_error = [
    [1.02e+01, 6.87e+00, 3.80e+00, 2.55e+00, 1.75e+00, 1.21e+00, 8.83e-01, 6.31e-01, 4.74e-01, 3.56e-01, 2.26e-01, 1.46e-01, 9.76e-02, 7.29e-02, 5.79e-02, 3.51e-02, 2.50e-02, 2.17e-02, 1.45e-02, 8.44e-03],# MM21
    [8.00e+00, 2.30e+00, 7.33e-01, 2.38e-01, 1.60e-01, 5.77e-02, 3.13e-02, 1.83e-02, 1.14e-02, 7.43e-03, 3.47e-03, 1.77e-03, 9.70e-04, 7.86e-04, 5.66e-04, 8.79e-05, 1.00e-04, 5.90e-04, 1.39e-05, 3.41e-05] # Ours
]


q1_performance = [
    [8.30e-01, 1.60e+00, 2.30e+00, 3.10e+00, 3.80e+00, 4.50e+00, 5.30e+00, 6.00e+00, 6.80e+00, 7.60e+00,
      9.90e+00, 1.06e+01, 1.18e+01, 1.35e+01, 1.49e+01, 1.87e+01, 2.18e+01, 2.47e+01, 2.80e+01, 3.11e+01], # MM21
    [9.00e-01, 1.70e+00, 2.40e+00, 3.20e+00, 3.90e+00, 4.70e+00, 5.40e+00, 6.20e+00, 6.90e+00, 7.70e+00,
      9.40e+00, 1.09e+01, 1.22e+01, 1.37e+01, 1.53e+01, 1.88e+01, 2.21e+01, 2.52e+01, 2.75e+01, 3.13e+01] # Ours
]

q1_error = [
    [6.40e+00, 2.50e+00, 1.30e+00, 6.70e-01, 3.90e-01, 2.40e-01, 1.60e-01, 1.10e-01, 8.10e-02, 6.10e-02,
      3.70e-02, 2.50e-02, 1.70e-02, 1.30e-02, 1.00e-02, 4.90e-03, 3.80e-03, 4.30e-03, 1.40e-03, 7.20e-04],# MM21
    [2.10e+00, 2.40e-01, 5.30e-02, 1.80e-02, 7.90e-03, 4.00e-03, 2.20e-03, 1.30e-03, 9.00e-04, 5.90e-04,
      3.00e-04, 1.40e-04, 7.30e-05, 8.40e-05, 7.80e-05, 1.50e-05, 1.5e-05, 1.00e-04, 2.40e-04, 2.60e-04] # Ours
]

q2_performance = [
    [6.60e+00, 1.25e+01, 1.92e+01, 2.57e+01, 3.26e+01, 3.95e+01, 4.50e+01, 5.24e+01, 6.08e+01, 6.24e+01,
      7.54e+01, 9.00e+01, 1.04e+02, 1.18e+02, 1.33e+02, 1.57e+02, 1.83e+02, 2.04e+02, 2.32e+02, 2.47e+02], # MM21
    [6.90e+00, 1.28e+01, 1.95e+01, 2.60e+01, 3.29e+01, 4.00e+01, 4.54e+01, 5.29e+01, 6.16e+01, 6.31e+01,
      7.62e+01, 9.06e+01, 1.04e+02, 1.19e+02, 1.34e+02, 1.59e+02, 1.86e+02, 2.07e+02, 2.34e+02, 2.50e+02] # Ours
]

q2_error = [
    [1.34e+01, 7.50e+00, 4.90e+00, 2.96e+00, 2.00e+00, 1.40e+00, 1.10e+00, 8.70e-01, 6.40e-01, 4.80e-01,
      2.70e-01, 1.60e-01, 1.00e-01, 6.50e-02, 4.60e-02, 2.50e-02, 1.90e-02, 1.40e-02, 9.00e-03, 4.60e-03],# MM21
    [1.34e+01, 5.90e+00, 5.10e+00, 1.60e+00, 6.40e-01, 2.30e-01, 9.50e-02, 4.80e-02, 2.50e-02, 1.40e-02,
      4.20e-03, 1.60e-03, 7.50e-04, 5.90e-04, 4.30e-04, 4.00e-05, 1.80e-05, 3.40e-04, 3.30e-05, 6.60e-05] # Ours
]


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

def display(axe, performance, error):
    name = ["[MM21]", "ours"]
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
display(axes[2], q1_performance, q1_error)
display(axes[3], q2_performance, q2_error)

titles = [
    r"P1 ($64 \times 16 \times 16$)",
    r"P2 ($32 \times 8 \times 8$)",
    r"Q1 ($64 \times 16 \times 16$)",
    r"Q2 ($32 \times 8 \times 8$)"
]

lines = [
    p1_error[0][len(p1_error[0]) -1],
    p2_error[0][len(p2_error[0]) -1],
    q1_error[0][len(q1_error[0]) -1],
    q2_error[0][len(q2_error[0]) -1],
]

print(lines)

i = 0
for ax in axes:
    #ax.axhline(y=lines[i], color='r', linestyle='--', linewidth=1)
    ax.grid(False)
    ax.set_yscale("log")
    if(i == 0 or i == 2):
        ax.set_ylabel("Erreur")
    ax.set_xlabel("coût (ms)")
    ax.legend()
    ax.set_title(titles[i])
    i = i + 1
    
# Echelles log
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("02_stable_constraint_full.png")
plt.show()
