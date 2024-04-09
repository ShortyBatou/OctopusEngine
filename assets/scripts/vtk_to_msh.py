import meshio
#path = "../result/vtk/Flexion/"
path = ""
file = "Torsion_Tetra10_12_4_4"

mesh = meshio.read(path+"{}.vtk".format(file))
mesh.write("{}.gmsh".format(file), file_format="gmsh")

