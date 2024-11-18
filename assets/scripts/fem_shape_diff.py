from sympy import *

x, y, z = symbols('x y z')


# P1
shape = [1 - x - y - z, x, y, z] 
L1 = shape[0]
L2 = shape[1]
L3 = shape[2]
L4 = shape[3]


# P2
shape = [ (2*L1 - 1)*L1, (2*L2 - 1)*L2, (2*L3 - 1)*L3, (2*L4 - 1)*L4, 4 * L1 * L2, 4 * L2 * L3, 4 * L1 * L3, 4 * L1 * L4, 4 * L2 * L4, 4 * L3 * L4] 

# P2 lumping
#shape = [L1*L1, L2*L2, L3*L3, L4*L4, 2 * L1 * L2, 2 * L2 * L3, 2 * L1 * L3, 2 * L1 * L4, 2 * L2 * L4, 2 * L3 * L4]



#Q2
'''
shape = [
    #corner nodes
    x*y*z*(x - 1)*(y - 1)*(z - 1)/8, 
    x*y*z*(x + 1)*(y - 1)*(z - 1)/8, 
    x*y*z*(x + 1)*(y + 1)*(z - 1)/8, 
    x*y*z*(x - 1)*(y + 1)*(z - 1)/8, 
    x*y*z*(x - 1)*(y - 1)*(z + 1)/8, 
    x*y*z*(x + 1)*(y - 1)*(z + 1)/8, 
    x*y*z*(x + 1)*(y + 1)*(z + 1)/8, 
    x*y*z*(x - 1)*(y + 1)*(z + 1)/8, 
    
    #edge
    -y*z*(x - 1)*(x + 1)*(y - 1)*(z - 1)/4, 
    -x*z*(x + 1)*(y - 1)*(y + 1)*(z - 1)/4, 
    -y*z*(x - 1)*(x + 1)*(y + 1)*(z - 1)/4, 
    -x*z*(x - 1)*(y - 1)*(y + 1)*(z - 1)/4, 
    -x*y*(x - 1)*(y - 1)*(z - 1)*(z + 1)/4, 
    -x*y*(x + 1)*(y - 1)*(z - 1)*(z + 1)/4, 
    -x*y*(x + 1)*(y + 1)*(z - 1)*(z + 1)/4, 
    -x*y*(x - 1)*(y + 1)*(z - 1)*(z + 1)/4, 
    -y*z*(x - 1)*(x + 1)*(y - 1)*(z + 1)/4, 
    -x*z*(x + 1)*(y - 1)*(y + 1)*(z + 1)/4, 
    -y*z*(x - 1)*(x + 1)*(y + 1)*(z + 1)/4, 
    -x*z*(x - 1)*(y - 1)*(y + 1)*(z + 1)/4, 
    
    #face
    z*(x - 1)*(x + 1)*(y - 1)*(y + 1)*(z - 1)/2, 
    y*(x - 1)*(x + 1)*(y - 1)*(z - 1)*(z + 1)/2, 
    x*(x + 1)*(y - 1)*(y + 1)*(z - 1)*(z + 1)/2,
    y*(x - 1)*(x + 1)*(y + 1)*(z - 1)*(z + 1)/2, 
    x*(x - 1)*(y - 1)*(y + 1)*(z - 1)*(z + 1)/2, 
    z*(x - 1)*(x + 1)*(y - 1)*(y + 1)*(z + 1)/2, 
    
    #volume
    -(x - 1)*(x + 1)*(y - 1)*(y + 1)*(z - 1)*(z + 1)]


'''

# P3
'''
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
d_diff = zeros(3, len(shape))
for i in range(len(shape)):
    d_diff[0,i] = diff(shape[i], x)
    d_diff[1,i] = diff(shape[i], y)
    d_diff[2,i] = diff(shape[i], z)
print(shape)
print(d_diff)

#print(d_shape_r0)
#print(d_shape_r1)
#print(d_shape_r2)
exit()

