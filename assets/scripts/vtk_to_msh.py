import meshio
#path = "../result/vtk/Flexion/"
path = ""
file = "Hexa_48_16_16_3x1x1"

mesh = meshio.read(path+"{}.vtk".format(file))
mesh.write("{}.gmsh".format(file), file_format="gmsh")

