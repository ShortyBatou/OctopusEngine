import meshio
#path = "../result/vtk/Flexion/"
path = ""
file = "Hexa_16_4_4_4x1x1"

mesh = meshio.read(path+"{}.vtk".format(file))
mesh.write("{}.gmsh".format(file), file_format="gmsh")

