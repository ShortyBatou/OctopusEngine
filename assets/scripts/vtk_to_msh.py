import meshio
#path = "../result/vtk/Flexion/"
path = ""
file = "Hexa_64_16_16"

mesh = meshio.read(path+"{}.vtk".format(file))
mesh.write("{}.gmsh".format(file), file_format="gmsh")

