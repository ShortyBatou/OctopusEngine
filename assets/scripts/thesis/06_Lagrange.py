import numpy as np
import matplotlib.pyplot as plt

def lagrange_basis(x_points, i, x):
    L = 1
    for j in range(len(x_points)):
        if j != i:
            L *= (x - x_points[j]) / (x_points[i] - x_points[j])
    return L

orders = ["Linéaire", "Quadratique", "Cubique"]
def plot_all_lagrange_orders(max_order=3):
    x_plot = np.linspace(-1.2, 1.2, 400)
    fig, axes = plt.subplots(1, max_order, figsize=(5 * max_order, 4))

    if max_order == 1:
        axes = [axes]

    for idx, order in enumerate(range(1, max_order + 1)):
        ax = axes[idx]
        x_points = np.linspace(-1, 1, order + 1)

        sum_plot = np.zeros_like(x_plot)

        color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9b59b6']  # stocker les couleurs de chaque courbe
        label_list = []  # stocker les labels

        for i in range(order + 1):
            y_plot = lagrange_basis(x_points, i, x_plot)
            ax.plot(x_plot, y_plot, label=f'$N_{i}$', color = color_list[i])
            sum_plot += y_plot
            label_list.append(f'$N_{i}(x)$')

        print(color_list)
        # Tracer la somme des fonctions de base
        ax.plot(x_plot, sum_plot, '--', color='red', linewidth=1)
        ax.axis([-1, 1, -0.5,1.5])
        ax.set_title(f'{orders[idx]}')
        ax.set_xlabel('x')
        ax.set_ylabel('$N_i$')
        ax.axhline(0, color='gray', linewidth=1)
        ax.grid(False)
        for i, (label, color) in enumerate(zip(label_list, color_list)):
            ax.text(
                x=0.5 - 0.15 * order * 0.5 + i * 0.15, y=0.85 ,
                s=label,
                color=color,
                fontsize=11,
                ha='center',
                transform=ax.transAxes
            )
            
        ax.text(
                x=0.075, y=0.8 ,
                s=r'$\sum_i N_i(x)$',
                color="red",
                fontsize=8,
                ha='center',
                transform=ax.transAxes
            )
    plt.suptitle('Polynômes de Lagrange (Ordres 1 à 3)', fontsize=16)
    plt.tight_layout(rect=[0, 0.0, 1, 0.9])
    plt.savefig('06_lagrange.png', dpi=200)
    plt.show()

# Exécution
plot_all_lagrange_orders()
