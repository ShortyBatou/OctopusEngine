import matplotlib.pyplot as plt
import sys

performance = [
    [2.5, 3, 5.5, 6.5, 7.5, 12.9, 19.3, 27], #Q1
    [8.3,15.9,18.9,22.1,41], #Q2
    [5,6.6,9.5,12.6,15.6,18,32], #P2
    [1.7,2.4,3,5,5.7,7.5,9,12,15,20.2,31.7]
]

error = [
    [1.4, 7.1e-1, 3.38e-1, 1.49e-1, 7.37e-2, 2.0e-2, 7.74e-3, 3.96e-3],#Q1
    [2.08e-1,2.33e-2,4.51e-3,1.47e-3,8.13e-4], #Q2
    [3.16e-1, 2.54e-2, 3.46e-3, 9.52e-4, 3.19e-4, 9.24e-5, 3.75e-6], #P2
    [1.41, 5.4e-1, 2.47e-1, 1.27e-1, 2.4e-1, 1e-1, 5.9e-2, 7.64e-2, 1.69e-2, 2.93e-2, 1.26e-2] #P1
]

names = [
    "Q1", "Q2", "P2", "P1"
]

# plot
colors = ["#2ecc71", "#1be0ca", "#9b59b6", "#3498db"]

fig, ax = plt.subplots()
for i in range(len(error)):
    col = colors[i % len(colors)]    
    ax.scatter(performance[i], error[i], label = names[i], color = col, linewidth=1.0)

#ax.plot([0, len(volumes[str(max_nb_planes)]) - 1], [sdf_volume, sdf_volume], "--", label = "SDF", color = "#119c24", linewidth=2.0)

ax.set_yscale('log')
ax.set_xlim((0, 30))
#ax.set_ylim((1, 2))
ax.set_xlabel("Cost (ms)", fontfamily = "Times New Roman", fontsize = 18)
ax.set_ylabel("Flexion Error", fontfamily = "Times New Roman", fontsize = 18)
ax.set_title("", fontfamily = "Times New Roman", fontsize = 18)
ax.legend()
plt.show()