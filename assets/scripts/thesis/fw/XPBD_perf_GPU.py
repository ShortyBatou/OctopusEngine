import numpy as np
import matplotlib.pyplot as plt

# Configuration de la figure : 2 lignes, 2 colonnes
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# Données
P2_nb = [20, 160, 540, 1280, 2500, 4320, 6860, 10240, 14580, 20000, 26620, 34560, 43940, 54880, 67500, 81920] 
P2_Base = [20.6, 42.8, 46.5, 58.1, 59.1, 65.8, 65.6, 69.7,71.9, 71.8, 75.3, 73.9, 77.2, 77.7, 81.9, 97.4]
P2_Opti = [17.0, 23.5, 23.3, 24.9, 25.5, 28.1, 29.5, 30.2, 30.7, 30.6, 31.1, 31.4, 36.4, 44.9, 49.4, 49.4] 

P3_nb = [20, 160, 540, 1280, 2500, 4320, 6860, 10240, 14580, 20000]
P3_Base = [34.7, 127.6, 149.8, 209.6, 219.9, 234.1, 245.1, 253.1, 261.5, 269.0]
P3_Opti = [12.5, 21.4, 29.7, 30.4, 30.8, 31.6, 30.4, 31.0, 34.3, 41.3]

Q1_nb = [4, 32, 108, 256, 500, 864, 1372, 2048, 2916, 4000, 5324, 6912, 8788, 10976, 13500, 16384, 19652]
Q1_Base = [7.6, 13.5, 14.2, 15.4, 16.8, 17.8, 16.4, 18.0, 19.9, 20.2, 20.3, 20.3, 20.3, 20.5, 20.9, 20.9, 24.0]
Q1_Opti = [6.7, 8.1, 7.4, 7.4, 8.2, 8.2, 8.3, 9.7, 10.5, 10.3, 10.0, 10.5, 11.3, 13.5, 14.4, 13.8, 14.4]

Q2_nb = [4, 32, 108, 256, 500, 864, 1372, 2048, 2916, 4000, 5324]
Q2_Base = [24.1, 86.2, 92.1, 121.7, 145.1, 145.2, 146.4, 153.1,178.8,188.4,195.5]
Q2_Opti = [5.4, 10.8, 10.0, 9.6, 10.9, 11.6, 11.5, 12.8,15.6,15.9,17.4]

def display_combined(ax, nb, base, opti, title, is_left=False, is_right=False, is_bottom=False, leg_loc='best'):
    colors = ["#3498db", "#8e44ad", "#27ae60"]
    x = np.array(nb)
    y_base = np.array(base)
    y_opti = np.array(opti)
    for i in range(len(base)):
      y_base[i] = y_base[i] / 100
      y_opti[i] = y_opti[i] / 100
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
