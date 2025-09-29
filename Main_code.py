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
from Simulation.Simulation import Simulation

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



#Define location of defects
yp = geometry.dy*((geometry.Ny)//2-1)
xp1 = 6*geometry.L/14 #(geometry.L/(geometry.Nx+1))*(4*(geometry.Nx)//10-7)
xp2 = 8*geometry.L/14 #(geometry.L/(geometry.Nx+1))*(4*(geometry.Nx)//10-7)

#an array of defects [x,y,[bx,by]]
defects=[
    [xp1,yp,[1.*pfcparms.a0,0]],
    [xp2,yp,[-1.*pfcparms.a0,0]]
    ]


#Define simulation parameters
simparams = SimParams(Csh=1, # Coefficient of sh energy in psi evolution
                      Cw          = 0,          # Coefficient of penalty term in psi evolution
                      penalty_Psi = True,       # is the penalty considerd in the evolution of psi ?
                      penalty_u   = True,       # is the penalty present in the definition of the elastic stress
                      SOLVE_MEC = False,
                      dt          = 1e-1,       #time step
                      tmax        = 1000,       #max simulation duration
                      outFreq     = 100,
                      L           = geometry.L, #domain length
                      H           = geometry.H) #domain height


Amp =  lambda avg,r : (1/5)*(np.absolute(avg)+(1/3)*np.sqrt(15*r-36*avg**2)) #ground state amplitudes of a hexagonal lattice
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

scale=0
g=0 if scale==0 else mechparams.mu/scale
filename  = "tt"+str(simparams.dt)+"_"+str(simparams.Cw)+"_"+str(scale)
# filename  = "debugRes"+str(simparams.dt)+"_"+str(simparams.Cw)+"_"+str(scale)
path = "./out/dipole/" # outputpath




simu = Simulation(pfcparms,simparams,geometry,mechparams,comm)


simu.set_IO(path,filename)

simu.initialize_Sim()


simu.initialize_PFC(defects)


def bottom(x):
    return np.isclose(x[1], 0)

bottom_dofs = fem.locate_dofs_geometrical(simu.mec_proc.mecFE.vector_sp2_quad, bottom)
bcs_u = [
    fem.dirichletbc(np.zeros((2,)), bottom_dofs, simu.mec_proc.mecFE.vector_sp2_quad)]

boundaries = [(1, lambda x: np.isclose(x[1], geometry.H))]

facet_indices, facet_markers = [], []
fdim = simu.domain.topology.dim - 1
for (marker, locator) in boundaries:
    facets = dolfinx.mesh.locate_entities(simu.domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = dolfinx.mesh.meshtags(simu.domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

ds = ufl.Measure("ds", domain = simu.domain, subdomain_data = facet_tag)

T = fem.Constant(simu.domain, dolfinx.default_scalar_type((g, 0)))

simu.initialize_FDM(bcs_u,[(T,ds(1))])
simu.write_output()


while simu.current_time < simparams.tmax:
    simu.current_time += simparams.dt
    simu.main_iteration()
    