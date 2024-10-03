from sympy import *
import numpy as np
from numpy.linalg import inv

order = 3

#generate the base functions
x, y, z = symbols('x y z')
dim = [x,y,z]
D = len(dim)
fn_prv = 1
fn = 3
result = [[1],[x],[y],[z]]
base = [[x],[y],[z]]
for o in range(order-1):
    print(base)
    new_base = [[],[],[]]
    for i in range(D):
        for j in range(i, D):
            for b in base[j]:
                new_base[i].append(b * dim[i])
    base = new_base
    result = result + new_base
    
basef = []
for r in result:
    for b in r:
        basef.append(b)
        
print("N = ", len(basef))        
print("Bases = ", basef)

#generate ref element's vertices
div = 1 / (order)
vertices = []
for i in range(order + 1):
    for j in range(order + 1):
        for k in range(order + 1):
            if(i+j+k < order+1):
                vertices.append([div * k, div*j, div*i])
print("N = ", len(vertices))        
print("Vertices = ", vertices)
N = len(vertices)

#compute the shape functions
mat_shape = np.ones((N,N))
for i in range(N):
    v = vertices[i]
    for j in range(1,N): 
        eval = basef[j].subs([(x, v[0]), (y, v[1]), (z,v[2])])
        mat_shape[i][j] = eval
mat_shape = inv(mat_shape)

mat_shape[abs(mat_shape) < 1e-8 ] = 0 #remove very small values due to float error

print("mat shape = ", mat_shape)
shapes = []
for i in range(N):
    shape = 0
    for j in range(N):
        shape += basef[j] * mat_shape[j][i]
    shape = nsimplify(shape) # eval trivial number opperation
    shape = factor(shape) # factorize
    shapes.append(shape) 
print("Shapes = ", shapes)
