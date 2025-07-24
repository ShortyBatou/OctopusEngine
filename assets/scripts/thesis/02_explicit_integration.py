import numpy as np
import matplotlib.pyplot as plt

# Fonction à intégrer et solution exacte
def f(t):
    return np.sin(t)

def exact_integral(t):
    return 1 - np.cos(t)

# Paramètres
t0 = 0
t_max = 4 * np.pi
dt = 1.
t_values = np.arange(t0, t_max + dt, dt)

# Euler explicite
F_euler = [0]
e_euler = [0]
for t in t_values[:-1]:
    F_next = F_euler[-1] + dt * f(t)
    F_euler.append(F_next)
    e_euler.append(F_next - exact_integral(t+dt))

# RK2 (Heun)
F_rk2 = [0]
e_rk2 = [0]
for t in t_values[:-1]:
    k1 = f(t)
    k2 = f(t + dt)
    F_next = F_rk2[-1] + 0.5 * dt * (k1 + k2)
    F_rk2.append(F_next)
    e_rk2.append(F_next - exact_integral(t+dt))

# RK4
F_rk4 = [0]
e_rk4 = [0]
for t in t_values[:-1]:
    k1 = f(t)
    k2 = f(t + dt / 2)
    k3 = f(t + dt / 2)
    k4 = f(t + dt)
    F_next = F_rk4[-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    F_rk4.append(F_next)
    e_rk4.append(F_next - exact_integral(t+dt))

# Solution exacte
t_exact = np.arange(t0, t_max + 0.01, 0.01)
F_exact = exact_integral(t_exact)

# Affichage
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

# Méthodes et couleurs
methods = ["Euler", "RK2", "RK4"]
results = [F_euler, F_rk2, F_rk4]
colors = ['red', 'green', 'blue']



for i in range(3):
    axes[0][i].plot(t_values, results[i], 'o-', color=colors[i])
    axes[0][i].plot(t_exact, F_exact, '--', color='black')
    axes[0][i].axis([t0, t_max, -0.25,2.25])
    axes[0][i].set_title(methods[i])
    axes[0][i].set_xlabel('t')
    axes[0][i].set_ylabel('∫ sin(t) dt')
    axes[0][i].grid(False)


results = [e_euler, e_rk2, e_rk4]
for i in range(0,3):
    axes[1][i].plot(t_values, results[i], 'o-', color='darkred')
    axes[1][i].axis([t0, t_max, -1,1])
    axes[1][i].set_xlabel('t')
    axes[1][i].set_ylabel('Erreur')
    axes[1][i].grid(False)


plt.tight_layout()
plt.savefig('02_explicit_integration.png', dpi=200)
plt.show()