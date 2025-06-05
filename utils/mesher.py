from dolfinx      import mesh
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio
# import gmsh 


def Get_RectangleMesh(comm,geom):
    return mesh.create_rectangle(comm, [(0.0, 0.0), (geom.L, geom.H)], [geom.Nx,geom.Ny],mesh.CellType.quadrilateral,np.float64)


def Get_Cube(comm, L,N):
    Domain = mesh.create_box(comm, [(0.0, 0.0,0.0), (100,100,100)], [20, 20,20],mesh.CellType.hexahedron)
    return Domain

# def create_mesh_gmsh(
#     L,
#     H,
#     res: float = 1,
#     wall_marker: int = 1,
# ):

#     gmsh.initialize()
#     if MPI.COMM_WORLD.rank == 0:
#         gmsh.model.add("MyDomain")

#         MyDomain = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
#         gmsh.model.occ.synchronize()
#         surfaces = gmsh.model.occ.getEntities(dim=1)
#         walls = []
#         for surface in surfaces:
#             # com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
#             walls.append(surface[1])
#         # Rotate channel theta degrees in the xy-plane


#         # Add physical markers
#         gmsh.model.addPhysicalGroup(2, [MyDomain], 2)
#         gmsh.model.setPhysicalName(2, 2, "bulk")

#         gmsh.model.addPhysicalGroup(1, walls, wall_marker)
#         gmsh.model.setPhysicalName(1, wall_marker, "Walls")

#         # Set number of threads used for mesh
#         gmsh.option.setNumber("Mesh.MaxNumThreads1D", MPI.COMM_WORLD.size)
#         gmsh.option.setNumber("Mesh.MaxNumThreads2D", MPI.COMM_WORLD.size)
#         gmsh.option.setNumber("Mesh.MaxNumThreads3D", MPI.COMM_WORLD.size)

#         # Set uniform mesh size
#         gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
#         gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)

#         # Generate mesh
#         gmsh.model.mesh.generate(2)
#     # Convert gmsh model to DOLFINx Mesh and meshtags
#     mesh, _, ft = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
#     gmsh.finalize()
#     return mesh, ft