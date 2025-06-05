import Mechanics
from Simulation.Parameters import *
from utils.mesher import *
from mpi4py import MPI
import dolfinx.io





comm = MPI.COMM_WORLD

geometry   = GeomParams(dx=1,dy=1,Nx=20,Ny=100)
domain = Get_RectangleMesh(comm,geometry)




f = fem.Constant(domain, np.array([0, -10.0],dtype=float))

simparams= SimParams(1,1,False,False,1e-2,1,geometry.L,geometry.H)

mechparams = MechParams(10*1e6,6*1e6,0,1,f,False,False,False,False)


filename="Tensile"

file = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./out/benchmark/"+filename+".xdmf", "w")
file.write_mesh(domain)
mec_proc = Mechanics.MecProc.MecProc(domain,mechparams,simparams,file)


def left(x):
    return np.isclose(x[0], 0)


def right(x):
    return np.isclose(x[0], geometry.L)


def bottom(x):
    return np.isclose(x[1], 0)


def top(x):
    return np.isclose(x[1], geometry.H)


bottom_dofs = fem.locate_dofs_geometrical(mec_proc.mecFE.vector_sp2_quad, bottom)
# left_dofs = fem.locate_dofs_geometrical(mec_proc.mecFE.vector_sp2_quad, left)
# right_dofs = fem.locate_dofs_geometrical(mec_proc.mecFE.vector_sp2_quad, right)

# # bcs = [
# #     fem.dirichletbc(np.zeros((2,)), left_dofs, mec_proc.mecFE.vector_sp2_quad),
# #     fem.dirichletbc(np.zeros((2,)), right_dofs, mec_proc.mecFE.vector_sp2_quad),
# # ]

# V_uy, mapping =  mec_proc.mecFE.vector_sp2_quad.sub(1).collapse()
# right_dofs_uy = fem.locate_dofs_geometrical(( mec_proc.mecFE.vector_sp2_quad.sub(1), V_uy), right)
# left_dofs_uy = fem.locate_dofs_geometrical(( mec_proc.mecFE.vector_sp2_quad.sub(1), V_uy), left)


V_uy, mapping =  mec_proc.mecFE.vector_sp2_quad.sub(1).collapse()
top_dofs_uy = fem.locate_dofs_geometrical(( mec_proc.mecFE.vector_sp2_quad.sub(1), V_uy), top)

# uD_y = fem.Function(V_uy)
# bcs = [
#     fem.dirichletbc(uD_y, left_dofs_uy,  mec_proc.mecFE.vector_sp2_quad.sub(1)),
#     fem.dirichletbc(uD_y, right_dofs_uy,  mec_proc.mecFE.vector_sp2_quad.sub(1)),
# ]

uD_y = fem.Function(V_uy)
uD_y.interpolate(lambda x : x[0]*0+1)
bcs = [
    fem.dirichletbc(np.zeros((2,)), bottom_dofs, mec_proc.mecFE.vector_sp2_quad),
    # fem.dirichletbc(uD_y, top_dofs_uy,  mec_proc.mecFE.vector_sp2_quad.sub(1)),
]





mec_proc.init_solver([],bcs)
mec_proc.ConfigureSolver_UPperp()
mec_proc.ConfigureSolver_u()

mec_proc.solveU()
mec_proc.extract_UE()
mec_proc.Get_Stress()
mec_proc.Get_Divergence()


mec_proc.write_output(0.0)

file.close()