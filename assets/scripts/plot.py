import matplotlib.pyplot as plt
import numpy as np
import math

#5it + Chebychev
a_error = [6.40E+04, 9.20E+03, 1.50E+03, 2.60E+02, 3.20E+01, 6.00E-02, 6.50E+00, 1.50E+01, 2.00E+01, 2.00E+01, 1.50E+01, 6.50E+00, 2.70E+00, 7.50E-02, 2.40E-04]
a_time = [2.60, 5.00, 7.40, 10.00, 12.00, 15.00, 18.00, 20.00, 23.00, 26.00, 31.00, 35.00, 40.00, 46.00, 49.00]
for i in range(len(a_error)):
    a_error[i] = math.sqrt(a_error[i])

#10it + Chebychev
b_error = [95000, 20000, 4500, 1100, 300, 74, 13, 0.8, 0.4, 1.6, 1.4, 0.1, 0.31, 0.96, 1.20]
b_time = [2.60, 5.00, 7.50, 9.80, 12.00, 15.00, 17.00, 20.00, 22, 25.00, 31.00, 35.00, 40.00, 46.0, 49.00]
for i in range(len(b_error)):
    b_error[i] = math.sqrt(b_error[i])
    
#Large dt + Chebychev    
c_error = [130000,85000,59000,44000,33000,26000,20500,17000,13000,11000,7800,5700,4200,3300,2500]
c_time = [2.60, 5.00, 7.50, 9.80, 12.00, 15.00, 17.00, 20.00,22, 25.00, 31.00, 35.00, 40.00, 46.0, 49.00]
for i in range(len(c_error)):
    c_error[i] = math.sqrt(c_error[i])
    
#Large dt 
f_error = [240000,170000,140000,110000,98000,97000,77000,69000,62000,56000,47000,39000,33000,28000,24000]
f_time = [2.60,5.00,7.30,9.60,12.00,14.00,17.00,19.00,22.00,24.00,30.00,34.00,38.00,43.00,47.00]
for i in range(len(f_error)):
    f_error[i] = math.sqrt(f_error[i])    

#small step 1 it
d_error = [26000,2500,500,144,64,20,12,20,1.2,0.25,17,11,32,35,6.50]
d_time = [2.60, 5.00, 7.50, 9.80, 12.00, 15.00, 17.00, 20.00,22, 25.00, 31.00, 35.00, 40.00, 46.0, 49.00]
for i in range(len(d_error)):
    d_error[i] = math.sqrt(d_error[i])

#small step 2 it
e_error = [57000, 8400, 1600, 450, 150,	55, 21, 8.8, 6.4, 4.2, 0.02, 0.01, 3.00, 0.9, 1.20]
e_time = [2.60, 5.00, 7.50, 9.80, 12.00, 15.00, 17.00, 20, 22, 25, 31, 35, 40, 46, 49]
for i in range(len(e_error)):
    e_error[i] = math.sqrt(e_error[i])


plt.grid()
plt.xlabel('cost (ms)', fontfamily = "Times New Roman", fontsize = 18)
plt.ylabel('MSE', fontfamily = "Times New Roman", fontsize = 18)
plt.title("VBD Convergence")
ax = plt.gca()
ax.set_yscale("symlog")
ax.set_ylim([0, 400])
ax.set_xlim([2, 50])
plt.plot(f_time, f_error, '#2c3e50', label = 'Base')
plt.plot(c_time, c_error, '#8e44ad', label = 'Base + Ch')
plt.plot(b_time, b_error, '#e74c3c', label = '10it + Ch')
plt.plot(a_time, a_error,'#f1c40f',  label = '5it + Ch')
plt.plot(e_time, e_error,'#16a085',  label = '2it')
plt.plot(d_time, d_error,'#3498db',  label = '1it')
plt.legend()
plt.savefig("VBD_Convergence.png", dpi=300)
plt.show()
