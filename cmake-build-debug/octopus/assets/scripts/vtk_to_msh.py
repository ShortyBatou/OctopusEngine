import meshio
#path = "../result/vtk/Flexion/"
path = ""
file = "Hexa_16_8_8_2x1x1"

mesh = meshio.read(path+"{}.vtk".format(file))
mesh.write("{}.gmsh".format(file), file_format="gmsh")

