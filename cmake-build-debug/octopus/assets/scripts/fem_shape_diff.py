from sympy import *

x, y, z = symbols('x y z')

#shape = [1 - x - y - z, x, y, z] # P1
shape = [(x+y+z-1)*(2*x+2*y+2*z-1), x*(2*x-1), y*(2*y-1), z*(2*z-1), -4*x*(x+y+z-1), 4*x*y, -4*y*(x+y+z-1),-4*z*(x+y+z-1),4*x*z, 4*y*z] # P2

''' # P3
a = 0.5
b = 9. / 2.
c = 27.
shape = [
    #corner nodes
    a * (3 * (1 - x - y - z) - 1) * (3 * (1 - x - y - z) - 2) * (1 - x - y - z),    
    a * (3 * x - 1) * (3 * x - 2) * x, 
    a * (3 * y - 1) * (3 * y - 2) * y,
    a * (3 * z - 1) * (3 * z - 2) * z,

    #/mid edge nodes
    b * (1 - x - y - z) * x * (3 * (1 - x - y - z) - 1),        
    b * (1 - x - y - z) * x * (3 * x - 1),
    b * x* y* (3 * x - 1), 
    b * x* y* (3 * y - 1),
    b * (1 - x - y - z) * y * (3 * y - 1),
    b * (1 - x - y - z) * y * (3 * (1 - x - y - z) - 1),
    b * (1 - x - y - z) * z * (3 * (1 - x - y - z) - 1),
    b * (1 - x - y - z) * z * (3 * z - 1), 
    b* x* z* (3 * x - 1), 
    b * x * z * (3 * z - 1), 
    b* y* z* (3 * y - 1), 
    b * y * z * (3 * z - 1),

    #mid face nodes
    c * (1 - x - y - z) * x * z,
    c * x * y * z, 
    c * (1 - x - y - z) * y * z, 
    c * (1 - x - y - z) * x * y  
]
'''
print(len(shape));
print(d_diff)

#print(d_shape_r0)
#print(d_shape_r1)
#print(d_shape_r2)
exit()

