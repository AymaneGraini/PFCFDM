import PhaseField.Blocked as Blocked
import PhaseField.Blocked.PfProc
import PhaseField.Blocked.PfComp
import Mechanics
from Simulation.Parameters import *
from Simulation.SimIO import *
from Simulation.crystals_db import *
from utils.mesher import *
from utils.monitor import *
from utils.utils import *
from mpi4py import MPI
import dolfinx.io
import time
import matplotlib.pyplot as plt
import ufl
from petsc4py import PETSc
import pyvista
from dolfinx.la import create_petsc_vector_wrap
from PFCproc_TODO.ProcessPFC_padFFT import *
from jax import vjp, jvp

if pyvista.OFF_SCREEN:
    pyvista.start_xvfb(wait=0.1)

comm = MPI.COMM_WORLD

pfcparms =  PfcParams(  a0         = 4*np.pi/np.sqrt(3),
                        qs         = hex_lat.qs,
                        ps         = hex_lat.ps,
                        r          = 1.4,
                        avg        = -0.5,
                        periodic   = False,
                        deg        = 4,
                        motion     = "up",
                        write_amps = False)



geometry   = GeomParams(dx=pfcparms.a0/7,
                        dy=np.sqrt(3)*pfcparms.a0/12,
                        Nx=7*50,  # the domain size should the multiple of 7 (or 3.5) for periodicity of e^(iq.x)
                        Ny=12*15) # the domain size should the multiple of 12 for periodicity of e^(iq.x)


simparams = SimParams(1,0.08,True,True,1e-1,1200,geometry.L,geometry.H)


filename  = "Annihilation"+str(simparams.dt)+"_"+str(simparams.Cw)
path = "./out/Annihilation/"
file = dolfinx.io.XDMFFile(MPI.COMM_WORLD, path+filename+".xdmf", "w")


domain = mesh.create_rectangle(comm, [(0.0, 0), (geometry.L, geometry.H)], [geometry.Nx,geometry.Ny],
                               mesh.CellType.quadrilateral,np.float64)
mesh_coords = domain.geometry.x
mapping     = np.lexsort((mesh_coords[:, 0], mesh_coords[:, 1]))
shape       = (geometry.Ny+1,geometry.Nx+1)

Proc = PFCProcessor(domain.geometry.x,DofMap=mapping,qs=pfcparms.qs,a0=pfcparms.a0,target_shape=shape,pads=(0,0))
f    = np.array([0, 0.0],dtype=float)
Amp =  lambda avg,r : (1/5)*(np.absolute(avg)+(1/3)*np.sqrt(15*r-36*avg**2)) # TODO this should not be here

mechparams = MechParams(  lambda_      = 3*Amp(pfcparms.avg,pfcparms.r)**2,
                          mu           = 3*Amp(pfcparms.avg,pfcparms.r)**2,
                          Cx           = 100*14*100/pfcparms.a0,
                          Cel          = 1,
                          f            = f,
                          periodic_UP  = False,
                          periodic_u   = False,
                          addNullspace = False)

mec_proc = Mechanics.MecProc.MecProc(domain,mechparams,simparams,file)
file.write_mesh(domain)

# SimulationNote = "Static simulation, Psi is first relaxed until t=200 then Ue is determined With the contribution of Q in the elastic stress. THen we gradient descent SH and penalty. throughout Dislocations are not moving, so we form alpha from psi then get Upperp to get UE (we considder the effect of Q in Ue)"
SimulationNote = "Static simulation without periodic Bcs of a fully coupled model"

write_sim_settings(path+filename+".json",SimulationNote,
                   **{
    "Geometry"  : geometry,
    "Mechanics" : mechparams,
    "PhaseField": pfcparms,
    "Simulation": simparams
})

yp = geometry.dy*((geometry.Ny)//2-1)
xp1 = (geometry.L/(geometry.Nx+1))*(4*(geometry.Nx)//10-7) # was 8-6 maybe useless
xp2 = (geometry.L/(geometry.Nx+1))*(6*(geometry.Nx)//10-1) # was 8-6 maybe useless

defects=[
    [xp1,yp,[1.*pfcparms.a0,0]],
    [xp2,yp,[-1.*pfcparms.a0,0]]
    ]




timestamps=[0]
SH_Energy = []

pfProc = Blocked.PfProc.PfProc(domain,pfcparms,simparams,file)
pfProc.Initialize_crystal(defects)
pfProc.init_solver()
pfProc.Configure_solver()
amps= jnp.array([Proc.C_Amp(jnp.array(pfProc.pfFe.psiout.x.array),i) for i in range(len(pfcparms.qs))])
pfProc.pfComp.update_cAmps(amps, Proc.rev_DofMap)
mec_proc.mecFE.Q.x.array[:]=Proc.Compute_Q(amps)


# pfProc.get_SH_Energy()
pfProc.Solve()
pfProc.Correct()
# file.close()
# exit()
t=0
n=0


t2=time.time()


print("Starting mecha")

pfProc.pfComp.Compute_alpha_tilde()
mec_proc.mecFE.alpha.x.array[:]=pfProc.pfComp.alphaT.x.array[:]*-1.0 #because alphaT from PFC is of negative sign
mec_proc.init_solver([],[])


mec_proc.ConfigureSolver_UPperp()
mec_proc.ConfigureSolver_u()
mec_proc.solveUPperp()
mec_proc.combine_UP()
mec_proc.solveU()
mec_proc.extract_UE()
mec_proc.compute_sym()
mec_proc.Get_Stress()
mec_proc.Get_Curls()

pfProc.write_output(t)

mec_proc.write_output(t)


component_errors = [
    error_L2(mec_proc.mecFE.UEsym.sub(i), mec_proc.mecComp.Qsym.sub(i))
    for i in range(4)
]
component_errors_rel = [
    error_L2_rel(mec_proc.mecFE.UEsym.sub(i), mec_proc.mecComp.Qsym.sub(i))
    for i in range(4)
]
erros_history=[component_errors]
mechanical_dissipation=[0]
rel_erros_history=[component_errors_rel]
timestamps=[t]
SH_Energy.append(pfProc.get_SH_Energy())
avg_history=[fem.assemble_scalar(pfProc.pfFe.Avg_form)]

dissipation = fem.form(ufl.inner(mec_proc.mecComp.sigmaUe,pfProc.pfComp.J)*pfProc.pfFe.dx)

while t<simparams.tmax:
    t+=simparams.dt
    pfProc.pfFe.dFQW.x.array[:] = jax_computegradFuq_sym(pfProc.pfFe.psiout.x.array,mec_proc.mecFE.UEsym.x.array,Proc)
    pfProc.Solve()
    pfProc.Correct()
    SH_Energy.append(pfProc.get_SH_Energy())
    amps= jnp.array([Proc.C_Amp(jnp.array(pfProc.pfFe.psiout.x.array),i) for i in range(len(pfcparms.qs))]) 
    pfProc.pfComp.update_cAmps(amps, Proc.rev_DofMap)
    pfProc.pfComp.Compute_current()
    pfProc.pfComp.Compute_alpha_tilde()
    mec_proc.mecFE.Q.x.array[:]=Proc.Compute_Q(amps)
    mec_proc.mecFE.alpha.x.array[:]=pfProc.pfComp.alphaT.x.array[:]*-1.0
    mec_proc.update_UP(pfProc)
    # mec_proc.solveUPperp()
    # mec_proc.combine_UP()
    mec_proc.solveU()
    mec_proc.extract_UE()  
    mec_proc.compute_sym()
    mec_proc.Get_Curls()
    mec_proc.Get_Stress()

    mechanical_dissipation.append(fem.assemble_scalar(dissipation))
    component_errors = [error_L2(mec_proc.mecFE.UEsym.sub(i), mec_proc.mecComp.Qsym.sub(i)) for i in range(4)]
    erros_history.append(component_errors)
    rel_erros_history.append([error_L2_rel(mec_proc.mecFE.UEsym.sub(i), mec_proc.mecComp.Qsym.sub(i)) for i in range(4)])
    avg_history.append(fem.assemble_scalar(pfProc.pfFe.Avg_form))

    timestamps.append(t)
    if n%20==0:
        pfProc.write_output(t)
        mec_proc.write_output(t)
    print("t= ", t) 
    if np.isnan(pfProc.pfFe.psiout.x.array).all():
        print("Psi diverged")
        break
    n+=1
# pfProc.write_output(t)
# mec_proc.write_output(t)


file.close()
np.savetxt(path+"Energy_"+filename+".csv",np.column_stack((timestamps,SH_Energy,mechanical_dissipation,avg_history)),delimiter="\t")
np.savetxt(path+"errors_"+filename+".csv",np.column_stack((timestamps,erros_history)),delimiter="\t")

