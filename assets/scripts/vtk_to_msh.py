import meshio
#path = "../result/vtk/Flexion/"
path = ""
file = "mesh"

mesh = meshio.read(path+"{}.msh".format(file))
mesh.write("{}.vtk".format(file), file_format="vtk")

