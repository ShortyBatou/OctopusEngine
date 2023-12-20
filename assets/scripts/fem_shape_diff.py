from sympy import *

x, y, z = symbols('x y z')

#shape = [1 - x - y - z, x, y, z]
shape = [(x+y+z-1)*(2*x+2*y+2*z-1), x*(2*x-1), y*(2*y-1), z*(2*z-1), -4*x*(x+y+z-1), 4*x*y, -4*y*(x+y+z-1),-4*z*(x+y+z-1),4*x*z, 4*y*z]
shape = [
            #corner nodes
            0.5 * (3*(1-x-y-z)-1) * (3*(1-x-y-z) - 2) * (1-x-y-z),
            0.5 * (3*x-1) * (3*x-2) * x,
            0.5 * (3*y-1) * (3*y-2) * y,
            0.5 * (3*z-1) * (3*z-2) * z,
            #mid edge nodes
            (9.0/2.0) * (1-x-y-z) * x * (3*x*(1-x-y-z)-1),
            (9.0/2.0) * (1-x-y-z) * x * (3*x-1),
            (9.0/2.0) * x*y*(3*x-1),
            (9.0/2.0) * x*y*(3*y-1),
            (9.0/2.0) * (1-x-y-z) * y * (3*y-1),
            (9.0/2.0) * (1-x-y-z) * y * (3*x*(1-x-y-z)-1),
            (9.0/2.0) * (1-x-y-z) * z * (3*x*(1-x-y-z)-1),
            (9.0/2.0) * x*z*(3*x-1),
            (9.0/2.0) * x*z*(3*y-1),
            (9.0/2.0) * (1-x-y-z) * z * (3*z-1),
            (9.0/2.0) * x*z*(3*z-1),
            (9.0/2.0) * y*z*(3*z-1),
            #mid face nodes
            27.0 * (1-x-y-z)*x*z,
            27.0 * x*y*z,
            27.0 * (1-x-y-z)*y*z,
            27.0 * (1-x-y-z)*x*z
        ]

print(len(shape));

d_diff = zeros(3, len(shape))
for i in range(len(shape)):
    d_diff[0,i] = diff(shape[i], x)
    d_diff[1,i] = diff(shape[i], y)
    d_diff[2,i] = diff(shape[i], z)

print(d_diff)

#print(d_shape_r0)
#print(d_shape_r1)
#print(d_shape_r2)