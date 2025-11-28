import numpy as np
import matplotlib.pyplot as plt

performance = [
    [0.4, 1.0, 1.3, 2.1, 2.4, 3.0, 3.3, 3.6, 3.8, 4.3, 4.5, 4.8, 6.0, 6.9, 7.9, 8.6, 11.2],  # P1
    [1.6, 4.2, 10.0],  # P2
    [0.6, 1.2, 1.7, 2.5, 2.8, 3.0, 3.9, 4.1, 4.3, 4.5, 4.8],  # Q1
    [7.5, 10.0, 12.0]  # Q2
]

size = [
    [1.41E+00,7.07E-01,4.71E-01,3.54E-01,2.83E-01,2.36E-01,2.02E-01,1.77E-01,1.57E-01,1.41E-01,1.29E-01,1.18E-01,1.09E-01,1.01E-01,9.43E-02,8.84E-02,8.32E-02],  # P1
    [1.41E+00,7.07E-01,4.71E-01],  # P2
    [1.73E+00,8.66E-01,5.77E-01,4.33E-01,3.46E-01,2.89E-01,2.47E-01,2.17E-01,1.92E-01,1.73E-01,1.57E-01],  # Q1
    [1.73E+00,8.66E-01,5.77E-01]  # Q2
]

error = [
    [7.50e-01, 1.50e-01, 4.20e-02, 1.50e-02, 7.10e-03, 3.30e-03, 1.70e-03, 1.00e-03, 7.50e-04, 6.10e-04, 5.40e-04, 3.90e-04, 2.60e-04, 1.40e-04, 1.10e-04, 7.89e-05, 4.96e-05],  # P1
    [3.50e-03, 1.06e-04, 3.80e-05],  # P2
    [1.60e-01, 2.40e-02, 6.00e-03, 2.10e-03, 8.70e-04, 4.30e-04, 2.70e-04, 1.60e-04, 6.40e-05, 5.30e-05, 2.25e-05],  # Q1
    [2.90e-03, 8.50e-05, 1.60e-05]  # Q2
]

fig, axes = plt.subplots(1, 1, figsize=(10, 6))

def display(axe, performance, error):
    name = ["P1", "P2", "Q1", "Q2"]
    colors = ["#3498db", "#8e44ad", "#62c467", "#16a085"]
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
        axe.scatter(x, y, color = colors[i])
        axe.plot(x, y, label=name[i], linewidth=2, color = colors[i], linestyle="-")
    axe.axhline(y=1e-4, color='r', linestyle='--', linewidth=1)
        #axe.plot(x_smooth, y_smooth, linewidth=2, color = colors[i], linestyle="-")

titles = ["Convergence pour le test de flexion", "GPU"]
display(axes, performance, error)
axes.set_xlim(0, 13)
axes.grid(False)
axes.set_yscale("log")
axes.set_ylabel("erreur", fontsize=16)
axes.set_xlabel("coût (ms)", fontsize=16)
axes.legend(fontsize=16)
axes.set_title(titles[0], fontsize=16)
"""
display(axes[1], size, error)
axes[1].set_xlim(5E-02, 2)
axes[1].set_ylim(1e-5, 1e-1)
axes[1].grid(False)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_ylabel("")
axes[1].set_xlabel("h")
axes[1].legend()
axes[1].set_title(titles[1])

L = 0
order = [2,3,3,4]
slope = [0.1,0.001,0.015,0.00008]
x = np.logspace(-2, 1, 200)
colors = ["#3498db", "#8e44ad", "#62c467", "#16a085"]
for i in range(len(slope)):
    axes[1].plot(x, slope[i]*x**(order[i]), '--', color=colors[i], label=r"$O(H^{%d})$" % (L+order[i]))
    """
# Echelles log
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("01_plot.png")
plt.show()
