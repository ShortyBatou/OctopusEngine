import numpy as np
import json
import matplotlib.pyplot as plt

import matplotlib.patches as patches
from matplotlib.path import Path


def get_rotation_error_data(file_path):
    f = open(file_path)
    data = json.load(f)
    distances = data["rotation_error"]["distance"]
    angles = data["rotation_error"]["angles"]
    n = len(distances) - 1
    for i in range(n): 
        for j in range(n):
            if(distances[j] > distances[j+1]):
                distances[j], distances[j+1] = distances[j+1], distances[j]
                angles[j], angles[j+1] = angles[j+1], angles[j]

    return distances, angles

def diff_with_ref(ref_dists, ref_angles, elem_dists, elem_angles):
    diff = []
    for i in range(len(elem_dists)):
        for j in range(len(ref_dists)):
            if(abs(elem_dists[i] - ref_dists[j]) < 1e-4):
                diff.append(abs(ref_angles[j] - elem_angles[i]))

    return diff

def diff_with_interpolation(elem_dists, elem_angles):
    diff = []
    for i in range(len(elem_dists)):
        lin_angle = elem_dists[i] * 180.0
        diff.append(abs(lin_angle - elem_angles[i]))
    return diff
'''
ref_path = '../result/vtk/Torsion/Torsion_Hexa_64_16_16_ref.json'
ref_distance, ref_angles = get_rotation_error_data(ref_path)


hexa_path = '../result/vtk/Torsion/Torsion_Hexa_16_4_4.json'
hexa_distance, hexa_angles = get_rotation_error_data(hexa_path)
'''
tetra_path = 'Torsion_Tetra_16_4_4.json'
tetra_distance, tetra_angles = get_rotation_error_data(tetra_path)

tetra10_path = 'Torsion_Tetra10_8_2_2.json'
tetra10_distance, tetra10_angles = get_rotation_error_data(tetra10_path)

tetra20_path = 'Torsion_Tetra20_4_2_2.json'
tetra20_distance, tetra20_angles = get_rotation_error_data(tetra20_path)

bad_tetra_path = 'Torsion_Bad_MeshTetra_16_4_4.json'
bad_tetra_distance, bad_tetra_angles = get_rotation_error_data(bad_tetra_path)

bad_tetra10_path = 'Torsion_Bad_MeshTetra10_8_2_2.json'
bad_tetra10_distance, bad_tetra10_angles = get_rotation_error_data(bad_tetra10_path)

bad_tetra20_path = 'Torsion_Bad_MeshTetra20_4_2_2.json'
bad_tetra20_distance, bad_tetra20_angles = get_rotation_error_data(bad_tetra20_path)

#tetra20_diff = diff_with_ref(ref_distance, ref_angles, tetra20_distance, tetra20_angles)
#tetra10_diff = diff_with_ref(ref_distance, ref_angles, tetra10_distance, tetra10_angles)
#tetra10_diff2 = diff_with_ref(ref_distance, ref_angles, tetra10_distance2, tetra10_angles2)
#tetra_diff = diff_with_ref(ref_distance, ref_angles, tetra_distance, tetra_angles)
#hexa_diff = diff_with_ref(ref_distance, ref_angles, hexa_distance, hexa_angles)

tetra_diff = diff_with_interpolation(tetra_distance, tetra_angles)
tetra10_diff = diff_with_interpolation(tetra10_distance, tetra10_angles)
tetra20_diff = diff_with_interpolation(tetra20_distance, tetra20_angles)

bad_tetra_diff = diff_with_interpolation(bad_tetra_distance, bad_tetra_angles)
bad_tetra10_diff = diff_with_interpolation(bad_tetra10_distance, bad_tetra10_angles)
bad_tetra20_diff = diff_with_interpolation(bad_tetra20_distance, bad_tetra20_angles)

#hexa_diff = diff_with_interpolation(hexa_distance, hexa_angles)
#prism_diff = diff_with_interpolation(prism_distance, prism_angles)
#pyramid_diff = diff_with_interpolation(pyramid_distance, pyramid_angles)
#ref_diff = diff_with_interpolation(ref_distance, ref_angles)

#plt.style.use('_mpl-gallery')
# make data

# plot
fig, ax = plt.subplots()
ax.grid()
ax.set(xlabel='Depth (%)', ylabel='Angle in degree', title='Non-biased Mesh')
ax.plot(tetra_distance, tetra_angles, '#e74c3c', label = 'P1')

ax.plot(tetra10_distance, tetra10_angles, '#c266e8', label = 'P2')
ax.plot(tetra20_distance, tetra20_angles, '#19e0b9', label = 'P3')
plt.legend()
fig.savefig("good_torsion.png", dpi=300)

fig2, ax2 = plt.subplots()
ax2.set(xlabel='Depth (%)', ylabel='Angle in degree', title='Biased Mesh')
ax2.grid()

ax2.plot(bad_tetra_distance, bad_tetra_angles, '#ab350a', label = 'P1')
ax2.plot(bad_tetra10_distance, bad_tetra10_angles, '#701196', label = 'P2')
ax2.plot(bad_tetra20_distance, bad_tetra20_angles, '#068c52', label = 'P3')
plt.legend()
fig2.savefig("bad_torsion.png", dpi=300)

fig3, ax3 = plt.subplots()
ax3.set(xlabel='Depth (%)', ylabel='Angle deviation from linear interpolation', title='')
ax3.grid()
ax3.plot(tetra_distance, tetra_diff, '#e74c3c', label = 'P1')
ax3.plot(tetra10_distance, tetra10_diff, '#c266e8', label = 'P2')
ax3.plot(tetra20_distance, tetra20_diff, '#19e0b9', label = 'P3')

ax3.plot(bad_tetra_distance, bad_tetra_diff, '#ab350a', label = 'P1 (biased)')
ax3.plot(bad_tetra10_distance, bad_tetra10_diff, '#701196', label = 'P2 (biased)')
ax3.plot(bad_tetra20_distance, bad_tetra20_diff, '#068c52', label = 'P3 (biased)')
plt.legend()
fig3.savefig("diff_linear_torsion.png", dpi=300)

plt.show()