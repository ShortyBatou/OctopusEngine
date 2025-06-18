from sympy import *
import numpy as np
from numpy.linalg import inv

order = 1

def build_tetra_base(dim, order):
    D = len(dim)
    result = [[1],[dim[0]],[dim[1]],[dim[2]]]
    base = [[dim[0]],[dim[1]],[dim[2]]]
    for o in range(order-1):
        print(base)
        new_base = [[],[],[]]
        for i in range(D):
            for j in range(i, D):
                for b in base[j]:
                    new_base[i].append(b * dim[i])
        base = new_base
        result = result + new_base
    return result

def sym_pow(x, n):
    r = 1
    for i in range(n):
        r = r * x
    return r

def build_hexa_base(dim, order):
    D = len(dim)
    x,y,z = dim[0], dim[1], dim[2]
    result = []
    for i in range(order+1):
        for j in range(order+1):
            for k in range(order+1):
                b = sym_pow(x, i) * sym_pow(y, j) * sym_pow(z, k)
                result.append([b])
    return result

#generate the base functions
x, y, z = symbols('x y z')
dim = [x,y,z]


#tetra
#result = build_tetra_base(dim, order)

#hexa
result = build_hexa_base(dim, order)
print(result)
basef = []
for r in result:
    for b in r:
        basef.append(b)
        
print("N = ", len(basef))        
print("Bases = ", basef)

#generate ref element's vertices
div = 1 / (order)
vertices = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]

#vertices = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
#vertices = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],
#            [0,-1,-1],[1,0,-1],[0,1,-1],[-1,0,-1],[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[0,-1,1],[1,0,1],[0,1,1],[-1,0,1],
#            [0,0,-1],[0,-1,0],[1,0,0],[0,1,0],[-1,0,0],[0,0,1],
#            [0,0,0]]


'''
for i in range(order + 1):
    for j in range(order + 1):
        for k in range(order + 1):
            if(i+j+k < order+1):
                vertices.append([div * k, div*j, div*i])'''
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
