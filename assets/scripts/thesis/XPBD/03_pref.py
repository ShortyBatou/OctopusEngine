import matplotlib.pyplot as plt
import numpy as np

elements = np.array([20, 160, 540, 1280, 2500, 4320, 6860, 10240, 14580, 20000])
cpu = np.array([19.8, 51.4, 63.1, 62.8, 65.6, 68.0, 66.3, 65.9, 65.4, 68.3])
gpu = np.array([1.1, 8.8, 30.3, 72.7, 146.8, 253.4, 404.8, 638.6, 980.4, 1394.0])

# Calcul du facteur d'accélération
speedup = cpu / gpu

# Recherche de l'intersection (speedup = 1)
intersection_x = None
intersection_y = None

for i in range(len(elements) - 1):
    if (speedup[i] - 1) * (speedup[i+1] - 1) < 0:  # changement de signe -> croisement
        # interpolation linéaire
        x0, x1 = elements[i], elements[i+1]
        y0, y1 = speedup[i], speedup[i+1]
        intersection_x = x0 + (1 - y0) * (x1 - x0) / (y1 - y0)
        intersection_y = 1

# Affichage du graphique
plt.figure(figsize=(8, 6))
plt.plot(elements, speedup, marker='o', color="green", label="Facteur d'accélération (CPU/GPU)")
plt.axhline(y=1, color="red", linestyle="--", label="Référence (1)")

if intersection_x is not None:
    plt.scatter(intersection_x, intersection_y, color="blue", zorder=5, label=f"Intersection ~ {intersection_x:.0f} éléments")

plt.xlabel("Nombre d'éléments")
plt.ylabel("Facteur d'accélération")
plt.title("Facteur d'accélération GPU par rapport au CPU")
plt.legend()
plt.grid(True)
plt.show()

if intersection_x is not None:
    print(f"⚡ Le GPU devient plus lent que le CPU à environ {intersection_x:.0f} éléments.")
else:
    print("Pas d'intersection trouvée : le GPU est toujours plus rapide ou toujours plus lent.")
