import PhaseField.Blocked as Blocked
import PhaseField.Blocked.PfProc
from Simulation.Parameters import *
from Simulation.crystals_db import *
from utils.mesher import *
from mpi4py import MPI
import dolfinx.io
import time
import matplotlib.pyplot as plt


pfcparms =  PfcParams(a0         = 4*np.pi/np.sqrt(3),
                    qs         = hex_lat.qs,
                    ps         = hex_lat.ps,
                    r          = 1.4,
                    avg        = -0.5,
                    periodic   = False,
                    deg        = 4,
                    motion     = "up",
                    write_amps = False)

comm = MPI.COMM_WORLD

geometry   = GeomParams(dx=pfcparms.a0/7,
                        dy=np.sqrt(3)*pfcparms.a0/12,
                        Nx=130, #102
                        Ny=133) #207*12//7

domain = mesh.create_rectangle(comm, [(0.0, 0), (geometry.L, geometry.H)], [geometry.Nx,geometry.Ny],
                               mesh.CellType.quadrilateral,np.float64)




simparams= SimParams(1,0,False,False,1e-1,10,geometry.L,geometry.H)



xp = geometry.L/2
yp = geometry.H/2

defects=[
    [xp,yp,[-1.*pfcparms.a0,0]],
    ]



filename="PFblockedtest"

file = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./out/benchmark/"+filename+".xdmf", "w")
file.write_mesh(domain)


pfProc = Blocked.PfProc.PfProc(domain,pfcparms,simparams,file)
pfProc.Initialize_crystal(defects)
avg_form = fem.form((1/(simparams.L*simparams.H))*pfProc.pfFe.psi0*pfProc.pfFe.dx)

pfProc.write_output(0.0)

pfProc.init_solver()
avgs=[fem.assemble_scalar(avg_form)]
print(avgs)
pfProc.Configure_solver()
print("Initing solver")
pfProc.Solve()
pfProc.Correct()
print(fem.assemble_scalar(avg_form))
# pfProc.write_output(1)

t=0
n=0
print('Solving t= ',t)
t1 =time.time()
while t<simparams.tmax:
    t+=simparams.dt
    pfProc.Solve()
    pfProc.Correct()
    pfProc.write_output(t)
    avgs.append(fem.assemble_scalar(avg_form))

    # print('now t= ',t)
    n+=1
t2 =time.time()
print("Blocked solved in ",t2-t1)
file.close()
# print(avgs)
# plt.plot(avgs)
# plt.show()