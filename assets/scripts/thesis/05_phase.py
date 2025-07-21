import matplotlib.pyplot as plt
import numpy as np

# Points caractéristiques
epsilon_Y = 0.01  # Déformation à la limite d'élasticité
epsilon_U = 0.08  # Déformation à la contrainte ultime
epsilon_F = 0.15  # Déformation à la rupture

sigma_Y = 250     # MPa, limite d'élasticité
sigma_U = 400     # MPa, contrainte ultime
sigma_F = 300     # MPa, contrainte à la rupture

# Déformation
eps = np.linspace(0, epsilon_F, 300)

# Contrainte (modèle simplifié en 3 segments)
sigma = np.piecewise(
    eps,
    [eps <= epsilon_Y,
     (eps > epsilon_Y) & (eps <= epsilon_U),
     (eps > epsilon_U)],
    [
        lambda e: sigma_Y / epsilon_Y * e,                          # élasticité
        lambda e: sigma_Y + (sigma_U - sigma_Y) * (e - epsilon_Y) / (epsilon_U - epsilon_Y),  # écrouissage
        lambda e: sigma_U - (sigma_U - sigma_F) * (e - epsilon_U) / (epsilon_F - epsilon_U)   # striction
    ]
)

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(eps, sigma, label="Courbe contrainte-déformation", color="navy", linewidth=2)

# Coloration des zones
plt.axvspan(0, epsilon_Y, facecolor='red', alpha=0.2, label="Élasticité")
plt.axvspan(epsilon_Y, epsilon_U, facecolor='yellowgreen', alpha=0.2, label="Écrouissage")
plt.axvspan(epsilon_U, epsilon_F, facecolor='skyblue', alpha=0.3, label="Striction / Rupture")

# Points clés
plt.plot(epsilon_Y, sigma_Y, 'ko')
plt.text(epsilon_Y, sigma_Y + 10, "Y", ha='center')
plt.plot(epsilon_U, sigma_U, 'ko')
plt.text(epsilon_U, sigma_U + 10, "U", ha='center')
plt.plot(epsilon_F, sigma_F, 'ko')
plt.text(epsilon_F, sigma_F + 10, "F", ha='center')

# Mise en forme
plt.title("Comportement d’un matériau ductile")
plt.xlabel("Déformation (ε)")
plt.ylabel("Contrainte (σ) [MPa]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
