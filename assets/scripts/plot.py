import matplotlib.pyplot as plt
import numpy as np


p1_time = [0.43, 0.85, 3.3, 6.6, 27, 56,110,240,520,1100]
p1_nbv = [20,36,81,153,425,825,1485,2673,5049,9537]

p2_time = [1.6,3.3,13,26,53,110,220,490]
p2_nbv = [81,153,425,765,1377,2673,5265,9945]

p3_time = [4.2,8.2,17,33,67,140,290,370]
p3_nbv = [208,364,637,1225,2275,4225,8281,9802]

plt.grid()
plt.xlabel('vertices')
plt.ylabel('time (ms)')
plt.title("Elements cost")

plt.plot(p1_nbv, p1_time,'#ab350a',  label = 'P1')
plt.plot(p2_nbv, p2_time, '#701196',  label = 'P2')
plt.plot(p3_nbv, p3_time, '#068c52',  label = 'P3')
plt.legend()
plt.show()