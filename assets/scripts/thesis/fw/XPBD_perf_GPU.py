import numpy as np
import matplotlib.pyplot as plt

# Configuration de la figure : 2 lignes, 2 colonnes
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# Données
P2_nb = [20, 160, 540, 1280, 2500, 4320, 6860, 10240, 14580, 20000, 26620, 34560, 43940, 54880, 67500, 81920] 
P2_Base = [36.7, 114, 123, 156, 160, 175, 178, 190, 198, 195, 205, 203, 208, 213, 264, 269]
P2_Opti = [24.9, 50.2, 50.0, 53.5, 53.2, 55.3, 59.5, 63.9, 67.2, 73.6, 83.6, 85.1, 98.4, 114.0, 132.0, 151.0] 

P3_nb = [20, 160, 540, 1280, 2500, 4320, 6860, 10240]
P3_Base = [80.36, 354, 415, 567, 589, 638, 656, 679]
P3_Opti = [23.8, 73.1, 75.8, 79.9, 82.9, 89.1, 89.7, 87.8]

Q1_nb = [4, 32, 108, 256, 500, 864, 1372, 2048, 2916, 4000, 5324, 6912, 8788, 10976, 13500, 16384]
Q1_Base = [14.9, 38.0, 39.5, 43.2, 47.0, 49.4, 45.6, 50.7, 55.17, 57.5, 56.6, 57.6, 57.8, 58.6, 58.7, 65.5]
Q1_Opti = [11.4, 20.0, 19.9, 20.1, 22.8, 22.2, 21.1, 24.6, 25.4, 29.1, 30.0, 32.2, 31.6, 36.3, 39.7, 39.4]

Q2_nb = [4, 32, 108, 256, 500, 864, 1372, 2048]
Q2_Base = [66.9, 254, 263, 341, 396, 413, 419, 453]
Q2_Opti = [9.1, 24.1, 24.7, 25.0, 27.1, 28.0, 28.4, 28.4]

def display_combined(ax, nb, base, opti, title, is_left=False, is_right=False, is_bottom=False, leg_loc='best'):
    colors = ["#3498db", "#8e44ad", "#27ae60"]
    x = np.array(nb)
    y_base = np.array(base)
    y_opti = np.array(opti)
    for i in range(len(base)):
      y_base[i] = y_base[i] / 200
      y_opti[i] = y_opti[i] / 200
    y_acc = y_base / y_opti

    # --- AXE GAUCHE : Coût ---
    ax.set_title(title, fontsize=17, fontweight='bold', pad=15)
    line1 = ax.plot(x, y_base, marker='o', label="Coût Base", color=colors[0], linewidth=2.5, markersize=6)
    line2 = ax.plot(x, y_opti, marker='o', label="Coût Opti", color=colors[1], linewidth=2.5, markersize=6)

    # Affichage conditionnel du TEXTE (Label) mais pas des VALEURS (Ticks)
    if is_left:
        ax.set_ylabel("Coût (ms)", fontsize=16, fontweight='bold')
    
    # On garde les graduations numériques sur tous les axes
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- AXE DROIT : Accélération ---
    ax_acc = ax.twinx()
    line3 = ax_acc.plot(x, y_acc, marker='^', label="Accélération", color=colors[2], linewidth=2.5, linestyle='--', markersize=7)

    if is_right:
        ax_acc.set_ylabel("Accélération (Base / Opti)", fontsize=16, color="#16ac40", fontweight='bold')
    
    ax_acc.tick_params(axis='y', labelcolor=colors[2], labelsize=16)

    if is_bottom:
        ax.set_xlabel("# éléments", fontsize=16, fontweight='bold')

    # --- GESTION DE LA LÉGENDE ---
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc=leg_loc, fontsize=12, frameon=True, framealpha=0.9)

# Configuration des 4 graphiques
# P2 : Gauche, Haut
display_combined(axes[0, 0], P2_nb, P2_Base, P2_Opti, "Performances P2", 
                 is_left=True, is_right=False, is_bottom=False, leg_loc='lower center')

# P3 : Droite, Haut
display_combined(axes[0, 1], P3_nb, P3_Base, P3_Opti, "Performances P3", 
                 is_left=False, is_right=True, is_bottom=False, leg_loc='center right')

# Q1 : Gauche, Bas
display_combined(axes[1, 0], Q1_nb, Q1_Base, Q1_Opti, "Performances Q1", 
                 is_left=True, is_right=False, is_bottom=True, leg_loc='lower center')

# Q2 : Droite, Bas
display_combined(axes[1, 1], Q2_nb, Q2_Base, Q2_Opti, "Performances Q2", 
                 is_left=False, is_right=True, is_bottom=True, leg_loc='center right')

# Ajustement des marges pour laisser de la place aux valeurs numériques
plt.subplots_adjust(hspace=0.28, wspace=0.25, left=0.08, right=0.92, top=0.92, bottom=0.08)

plt.savefig("XPBD_perf_GPU.png", bbox_inches='tight', dpi=300)
plt.show()
