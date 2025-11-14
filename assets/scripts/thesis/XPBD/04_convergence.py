import numpy as np
import matplotlib.pyplot as plt

CPU_performance = [
    [0.47, 5.8, 15.9, 31.1, 81.5, 407, 1129], #P1
    [2.7 , 35.9, 82.5], #P2
    [0.041, 0.53, 6.1, 16.1, 34.9, 100.3, 500], #Q1
    [0.8, 15.9] #Q2
]

GPU_performance = [
    [3, 5.2, 7.4, 8.8, 8.9, 10.8, 12.1, 14], #P1
    [3, 11.6, 16.7], #P2
    [0.35, 1.6, 2.5, 3.2, 3.7, 4.6, 6.3], #Q1
    [4.4, 28.3] #Q2
]

CPU_error = [
    [1.43e-1, 1.45e-2, 6.5e-3, 3.5e-3, 1.6e-3, 5.2e-4, 6.26e-4], #P1
    [3.6e-3,3.3e-4,5.7e-5], #P2
    [1.6e-1,2.08e-2,1.1e-3,6.2e-4,3.6e-4,2.0e-4,3.1e-5], #Q1
    [2.3e-3, 2.5e-5] #Q2
]

GPU_error = [
    [1.43e-1, 1.45e-2, 6.5e-3, 3.5e-3, 1.6e-3, 5.2e-4, 3.78e-4, 6.26e-4], #P1
    [3.6e-3,3.3e-4,5.7e-5], #P2
    [1.6e-1,2.08e-2,1.1e-3,6.2e-4,3.6e-4,2.0e-4,3.1e-5], #Q1
    [2.3e-3, 2.5e-5] #Q2
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes = axes.flatten()

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

display(axes[0], CPU_performance, CPU_error)
axes[0].set_xlim(-10, 600)
display(axes[1], GPU_performance, GPU_error)
axes[1].set_xlim(-0.5, 30)
titles = ["CPU", "GPU"]
i = 0
for ax in axes:
    ax.grid(False)
    ax.set_yscale("log")
    if(i == 0):
        ax.set_ylabel("erreur")
    ax.set_xlabel("coût (ms)")
    ax.legend()
    ax.set_title(titles[i])
    i = i + 1
    
# Echelles log
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("01_plot.png")
plt.show()
