import matplotlib.pyplot as plt
import numpy as np

p1_time = [0,0.21, 1.7, 13,42, 106,213,307,604,883,1282,1758,2035]
p1_memory = [29,29,29,30,31,33,36,42,49,59,71,87,95]
p1_nbv = [0,8,27,125,343,729,1331,2197,3375,4913,6859,9261,10648]
p1_nbe = [0,6,48,384,1296,3072,6000,10368,16464,24576,34992,48000,55566]


p2_time = [0,0.83,6.5,21.1,50.2,100,174,287,426,616,846,1133]
p2_memory = [29,29,30,30,31,32,35,38,42,46,53,60]
p2_nbv = [0,27,125,343,729,1331,2197,3375,4913,6859,9261,12167]
p2_nbe = [0,6,48,162,384,750,1296,2058,3072,4374,6000,7986]


p3_time = [0,2.3,17.7,57.8,137,277,483,772]
p3_memory = [29,29,30,31,33,37,43,50]
p3_nbv = [0,64,343,1000,2197,4096,6859,10648]
p3_nbe = [0,6,48,162,384,750,1296,2058]

plt.grid()
plt.xlabel('vertices')
plt.ylabel('time (ms)')
plt.title("a")
ax = plt.gca()
ax.set_xlim([0, 10000])

plt.plot(p1_nbv, p1_time,'#ab350a',  label = 'P1')
plt.plot(p2_nbv, p2_time, '#701196',  label = 'P2')
plt.plot(p3_nbv, p3_time, '#068c52',  label = 'P3')
plt.legend()
plt.show()
plt.savefig("elem_performance.png", dpi=300)

plt.grid()
plt.xlabel('vertices')
plt.ylabel('memory (mo)')
plt.title("z")
ax = plt.gca()
ax.set_xlim([0, 10000])

plt.plot(p1_nbv, p1_memory,'#ab350a',  label = 'P1')
plt.plot(p2_nbv, p2_memory, '#701196',  label = 'P2')
plt.plot(p3_nbv, p3_memory, '#068c52',  label = 'P3')
plt.legend()
plt.show()
plt.savefig("elem_memory.png", dpi=300)