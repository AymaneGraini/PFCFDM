#Import the Blocked solver for the phase field part, we can chose between a Blocked or Blocked formulation
import PhaseField.Blocked as Blocked
import PhaseField.Blocked.PfProc
import PhaseField.Blocked.PfComp

#Import the Mechanics solver
import Mechanics
# import dataclasses for simulation parameters
from Simulation.Parameters import *
from Simulation.SimIO import * #writer json file for parameters
from Simulation.crystals_db import * #import crystal database

#import solver monitor for residuals and convergence
from utils.monitor import *
#import different utility functions
from utils.utils import *
#import MPI
from mpi4py import MPI
#import IO from dolfinx for xdmf files
import dolfinx.io
# import external processor for the phase field
from PFCproc_TODO.ProcessPFC import *
from petsc4py import PETSc

import time
###################################################
###################################################

#initialize a MPI communicator
comm = MPI.COMM_WORLD



# Define PFC parameters
pfcparms =  PfcParams(  a0         = 4*np.pi/np.sqrt(3), #lattice spacing
                        qs                 = hex_lat.qs, #array of 1st mode wave vectors
                        ps                 = hex_lat.ps, #array of 2nd mode wave vectors
                        r                  = 0.8,        # cooling parameter
                        avg                = -0.43,       #Target average
                        periodic           = False,      #wether periodic bcs are used or not
                        deg                = 4,          #pde degree 4 for uncoserved, 6 = conserverd
                        motion             = "J",        # how do we compute the current
                        ConservationMethod = "LM",       # how do we conserve the average
                        write_amps         = False)      #write amps ?


#Define geometry parameters
geometry   = GeomParams(dx=pfcparms.a0/7,
                        dy=np.sqrt(3)*pfcparms.a0/12,
                        Nx=7*70,  # the domain size should the multiple of 7 (or 3.5) for periodicity of e^(iq.x)
                        Ny=12*15) # the domain size should the multiple of 12 for periodicity of e^(iq.x)

#Define simulation parameters
simparams = SimParams(Csh=1, # Coefficient of sh energy in psi evolution
                      Cw          = 0.01,          # Coefficient of penalty term in psi evolution
                      penalty_Psi = True,       # is the penalty considerd in the evolution of psi ?
                      penalty_u   = True,       # is the penalty present in the definition of the elastic stress
                      SOLVE_MEC   = True,
                      dt          = 1e-1,       #time step
                      tmax        = 1000,       #max simulation duration
                      outFreq     = 100,
                      L           = geometry.L, #domain length
                      H           = geometry.H) #domain height


Amp =  lambda avg,r : (1/5)*(np.absolute(avg)+(1/3)*np.sqrt(15*r-36*avg**2)) #ground state amplitudes of a hexagonal lattice
print("The amplitude is ", Amp(pfcparms.avg,pfcparms.r))
#Body force array
f    = np.array([0, 0.0],dtype=float)
#Define Mechanical parameters
mechparams = MechParams(  lambda_      = 3*Amp(pfcparms.avg,pfcparms.r)**2, #lamé 1st coeff
                          mu           = 3*Amp(pfcparms.avg,pfcparms.r)**2, # Lamé 2nd coeff
                          Cx           = 100*14*100/pfcparms.a0, # boundary term penalty weight in div-curl
                          Cel          = 1, # weight of elastic energy
                          f            = f, # body force
                          periodic_UP  = False,
                          periodic_u   = False,
                          addNullspace = False)

#SIMULATIONFILE name

scale=20
g=0 if scale==0 else mechparams.mu/scale
filename  = "Pen"+str(simparams.dt)+"_"+str(simparams.Cw)+"_"+str(scale)
# filename  = "debugRes"+str(simparams.dt)+"_"+str(simparams.Cw)+"_"+str(scale)
path = "./out/dipole/" # outputpath
file = dolfinx.io.XDMFFile(MPI.COMM_WORLD, path+filename+".xdmf", "w") #XDMF File for output

#Build a rectangular domain of size LxH with Nx cells in x and Ny in y direction
domain = mesh.create_rectangle(comm, [(0.0, 0), (geometry.L, geometry.H)], [geometry.Nx,geometry.Ny],
                               mesh.CellType.quadrilateral,np.float64)

#write the mesh into the file
file.write_mesh(domain)


#Extract mesh coordinates
mesh_coords = domain.geometry.x
mapping     = np.lexsort((mesh_coords[:, 0], mesh_coords[:, 1])) #sort them by X then by Y and save the array
shape       = (geometry.Ny+1,geometry.Nx+1) # Target mesh grid shape for scalar ndarrays

#PFC external processor
ProcEXT = PFCProcessorEXT(domain.geometry.x,DofMap=mapping,qs=pfcparms.qs,a0=pfcparms.a0,target_shape=shape,pads=(0,0))







print("Shear modulus is ",mechparams.mu)

# exit()
#Define an FE mechanical processor with those parameter
mec_proc = Mechanics.MecProc.MecProc(domain,mechparams,simparams,file)

 
#SImulation note for json output
SimulationNote = "Annihilation of a dislocation dipole of opposite sign"
#write all parameters into json file
write_sim_settings(path+filename+".json",SimulationNote,
                   **{
    "Geometry"  : geometry,
    "Mechanics" : mechparams,
    "PhaseField": pfcparms,
    "Simulation": simparams
})

#Define location of defects
yp = geometry.dy*((geometry.Ny)//2-1)
xp1 = 6*geometry.L/14 #(geometry.L/(geometry.Nx+1))*(4*(geometry.Nx)//10-7)
xp2 = 8*geometry.L/14 #(geometry.L/(geometry.Nx+1))*(4*(geometry.Nx)//10-7)

#an array of defects [x,y,[bx,by]]
defects=[
    [xp1,yp,[1.*pfcparms.a0,0]],
    [xp2,yp,[-1.*pfcparms.a0,0]]
    ]


t=0 #time
n=0 #number of iteration


timestamps=[t]

SH_Energy = []

#Define a Phasefield FE processor
pfProc = Blocked.PfProc.PfProc(domain,pfcparms,simparams,file)
#Initialize psi with a defected crystal using defect array
pfProc.Initialize_crystal(defects)

# Initialize PFC solver and create the required forms
pfProc.init_solver()
#Configure the solver
pfProc.Configure_solver()

#Compute the complex amplitudes



#perform 1 solve to get a ground state crystal
pfProc.Solve()
#correct if needed the average and update previous values (t=0)
pfProc.Correct()

#Append the current energy in SH array
SH_Energy.append(pfProc.get_SH_Energy())


amps= jnp.array([ProcEXT.C_Amp(jnp.array(pfProc.pfFe.psiout.x.array),i) for i in range(len(pfcparms.qs))])
#update the complex amplitude functions
pfProc.pfComp.update_cAmps(amps, ProcEXT.rev_DofMap)

mec_proc.mecFE.Q.x.array[:]=ProcEXT.Compute_Q(amps)


mec_proc.Get_Curls()        # Compute curls

mec_proc.mecFE.alpha.interpolate(mec_proc.mecComp.curlQ)
#Intialize alpha of mechanics



print("Starting mecha")

def bottom(x):
    return np.isclose(x[1], 0)

bottom_dofs = fem.locate_dofs_geometrical(mec_proc.mecFE.vector_sp2_quad, bottom)
bcs_u = [
    fem.dirichletbc(np.zeros((2,)), bottom_dofs, mec_proc.mecFE.vector_sp2_quad)]

boundaries = [(1, lambda x: np.isclose(x[1], geometry.H))]

facet_indices, facet_markers = [], []
fdim = domain.topology.dim - 1
for (marker, locator) in boundaries:
    facets = dolfinx.mesh.locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = dolfinx.mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

ds = ufl.Measure("ds", domain = domain, subdomain_data = facet_tag)

T = fem.Constant(domain, dolfinx.default_scalar_type((g, 0)))

#Initialize mechanical solver without Dirichelt bcs
mec_proc.init_solver([],bcs_u,[(T,ds(1))])

#configure solverss
mec_proc.ConfigureSolver_UPperp()
mec_proc.ConfigureSolver_u()
mec_proc.Configure_solver_zp()

#Solve the div-curl system to get UpPerp
mec_proc.solveUPperp()
# mec_proc.solveUPperp()
# exit()
# mec_proc.init_UP_para(defects,h=pfcparms.a0/2)

mec_proc.combine_UP()       # build UP=UpPerp+UpPara

mec_proc.solveU()           # Solve mechanical equilibrium

mec_proc.extract_UE()       # Extract elastic distortion


mec_proc.compute_sym()      # Compute symmetric parts of UE and Q
mec_proc.Get_Stress()       # Calculate stresses

mec_proc.Get_Curls()        # Compute curls

#Write output
pfProc.write_output(t)
mec_proc.write_output(t)
# file.close()
# exit()

#Compute absolute errors
component_errors = [
    error_L2(mec_proc.mecFE.UEsym.sub(i), mec_proc.mecComp.Qsym.sub(i))
    for i in range(4)
]
#compute relative errors
component_errors_rel = [
    error_L2_rel(mec_proc.mecFE.UEsym.sub(i), mec_proc.mecComp.Qsym.sub(i))
    for i in range(4)
]

#Append errors to arrays
erros_history=[component_errors]
rel_erros_history=[component_errors_rel]

#initialize mechanical dissipation to 0
mechanical_dissipation=[0]

#append the average
avg_history=[fem.assemble_scalar(pfProc.pfFe.Avg_form)]

#compile the form of mechanical disspation
dissipation = fem.form(ufl.inner(mec_proc.mecComp.sigmaUe,pfProc.pfComp.J)*pfProc.pfFe.dx)

# -----------------------------
# Main simulation loop
# -----------------------------
while t < simparams.tmax:
    t += simparams.dt

    # -----------------------------
    # Phase field evolution
    # -----------------------------

    # compute penalty term using previous step's configuration
    if simparams.Cw>0:
        pfProc.pfFe.dFQW.x.array[:] = jax_computegradFuq(
            pfProc.pfFe.psiout.x.array,
            mec_proc.mecFE.UE.x.array,
            ProcEXT
        )

    pfProc.Solve()    # Solve phase field evolution equation
    pfProc.Correct()  # Apply average correction, update output, and overwrite old solution

    # Compute complex amplitudes and update associated quantities
    amps = jnp.array([
        ProcEXT.C_Amp(jnp.array(pfProc.pfFe.psiout.x.array), i)
        for i in range(len(pfcparms.qs))
    ])
    pfProc.pfComp.update_cAmps(amps, ProcEXT.rev_DofMap)
    pfProc.pfComp.Compute_current()      # Topological charge current
    # pfProc.pfComp.Compute_alpha_tilde()  # Dislocation density tensor

    # -----------------------------
    # Mechanics update
    # -----------------------------

    mec_proc.mecFE.Q.x.array[:] = ProcEXT.Compute_Q(amps)  # Configurational distortion
    mec_proc.Get_Curls()        # Compute curls
    mec_proc.mecFE.alpha.interpolate(mec_proc.mecComp.curlQ)

    mec_proc.solveUPperp() # gived chi_p or up_perp

    mec_proc.mecFE.J.interpolate(pfProc.pfComp.J)
    mec_proc.solve_Up_para() # given chi_p or up_perp
    mec_proc.combine_UP()
    mec_proc.solveU()           # Solve mechanical equilibrium
    mec_proc.extract_UE()       # Extract elastic distortion

    mec_proc.compute_sym()      # Compute symmetric parts of UE and Q
    mec_proc.Get_Stress()       # Calculate stresses
    mec_proc.Get_Curls()        # Compute curls
    mec_proc.mecFE.alpha.interpolate(mec_proc.mecComp.curlQ)

    # # # -----------------------------
    # # # indicators for monitoring
    # # # -----------------------------

    # timestamps.append(t)
    # SH_Energy.append(pfProc.get_SH_Energy())
    # mechanical_dissipation.append(fem.assemble_scalar(dissipation))

    # # Compute error metrics
    # component_errors = [
    #     error_L2(mec_proc.mecFE.UEsym.sub(i), mec_proc.mecComp.Qsym.sub(i))
    #     for i in range(4)
    # ]
    # erros_history.append(component_errors)

    # rel_erros_history.append([
    #     error_L2_rel(mec_proc.mecFE.UEsym.sub(i), mec_proc.mecComp.Qsym.sub(i))
    #     for i in range(4)
    # ])
    # avg_history.append(fem.assemble_scalar(pfProc.pfFe.Avg_form))

    # # Periodic output writing
    if n % simparams.outFreq == 0:
        pfProc.write_output(t)
        mec_proc.write_output(t)

    # Divergence check
    if np.isnan(pfProc.pfFe.psiout.x.array).all():
        print("Psi diverged")
        break
    
    # if np.isclose(t,1):
    #     break

    print("Current time = ", t)
    n += 1
    # break

#Close file
file.close()

#write indicators
np.savetxt(path+"Energy_"+filename+".csv",np.column_stack((timestamps,SH_Energy,mechanical_dissipation,avg_history)),delimiter="\t")
np.savetxt(path+"errors_"+filename+".csv",np.column_stack((timestamps,erros_history)),delimiter="\t")

