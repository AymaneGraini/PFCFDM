
import PhaseField.Blocked as Blocked
import PhaseField.Blocked.PfProc
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


class Simulation:
    def __init__(self,pfcparms,simparams,geometry,mechparams,comm):

        self.pfcparms = pfcparms
        self.simparams = simparams
        self.geometry = geometry
        self.mechparams = mechparams

        self.current_time = 0
        self.average_history=[]
        self.errors_history=[]
        self.rel_erros_history=[]
        self.avg_history=[]
        self.mechanical_dissipation=[]
        self.t = 0
        self.COMM = comm


    def set_IO(self,path,filename):
        self.file = dolfinx.io.XDMFFile(MPI.COMM_WORLD, path+filename+".xdmf", "w") #XDMF File for output

    
    def initialize_Sim(self):
        self.domain = mesh.create_rectangle(self.COMM, [(0.0, 0), (self.geometry.L, self.geometry.H)], [self.geometry.Nx,self.geometry.Ny],
                               mesh.CellType.quadrilateral,np.float64)
        
        #Extract mesh coordinates
        mesh_coords = self.domain.geometry.x
        mapping     = np.lexsort((mesh_coords[:, 0], mesh_coords[:, 1])) #sort them by X then by Y and save the array
        shape       = (self.geometry.Ny+1,self.geometry.Nx+1) # Target mesh grid shape for scalar ndarrays

        self.ProcEXT = PFCProcessorEXT(self.domain.geometry.x,DofMap=mapping,qs=self.pfcparms.qs,a0=self.pfcparms.a0,target_shape=shape,pads=(0,0))
        self.mec_proc = Mechanics.MecProc.MecProc(self.domain,self.mechparams,self.simparams,self.file)
        self.pfProc = Blocked.PfProc.PfProc(self.domain,self.pfcparms,self.simparams,self.file)


        self.file.write_mesh(self.domain)


    def initialize_PFC(self, defects):
        self.pfProc.Initialize_crystal(defects)

        # Initialize PFC solver and create the required forms
        self.pfProc.init_solver()
        #Configure the solver
        self.pfProc.Configure_solver()

        #Compute the complex amplitudes
        amps= jnp.array([self.ProcEXT.C_Amp(jnp.array(self.pfProc.pfFe.psiout.x.array),i) for i in range(len(self.pfcparms.qs))])
        #update the complex amplitude functions
        self.pfProc.pfComp.update_cAmps(amps, self.ProcEXT.rev_DofMap)

        #perform 1 solve to get a ground state crystal
        self.pfProc.Solve()
        #correct if needed the average and update previous values (t=0)
        self.pfProc.Correct()

    def initialize_FDM(self,bcsD,bcsN):
         
        self.mec_proc.Get_Curls()        # Compute curls

        self.mec_proc.mecFE.alpha.interpolate(self.mec_proc.mecComp.curlQ) # Initialize alpha in FDM from PFC

        #Initialize mechanical solver without Dirichelt bcs
        self.mec_proc.init_solver([],bcsD,bcsN)

        #configure solverss
        self.mec_proc.ConfigureSolver_UPperp()
        self.mec_proc.ConfigureSolver_u()
        self.mec_proc.Configure_solver_zp()

        #Solve the div-curl system to get UpPerp
        self.mec_proc.solveUPperp()

        self.mec_proc.combine_UP()       # build UP=UpPerp+UpPara

        self.mec_proc.solveU()           # Solve mechanical equilibrium

        self.mec_proc.extract_UE()       # Extract elastic distortion


        self.mec_proc.compute_sym()      # Compute symmetric parts of UE and Q
        self.mec_proc.Get_Stress()       # Calculate stresses

        self.mec_proc.Get_Curls()        # Compute curls

    def write_output(self):
        self.pfProc.write_output(self.current_time)
        self.mec_proc.write_output(self.current_time)

         
    def main_iteration(self):
        # -----------------------------
        # Phase field evolution
        # -----------------------------

        # compute penalty term using previous step's configuration
        if self.simparams.Cw>0:
            self.pfProc.pfFe.dFQW.x.array[:] = jax_computegradFuq(
                self.pfProc.pfFe.psiout.x.array,
                self.mec_proc.mecFE.UE.x.array,
                self.ProcEXT
            )

        self.pfProc.Solve()    # Solve phase field evolution equation
        self.pfProc.Correct()  # Apply average correction, update output, and overwrite old solution

        # Compute complex amplitudes and update associated quantities
        amps = jnp.array([
            self.ProcEXT.C_Amp(jnp.array(self.pfProc.pfFe.psiout.x.array), i)
            for i in range(len(self.pfcparms.qs))
        ])
        self.pfProc.pfComp.update_cAmps(amps, self.ProcEXT.rev_DofMap)
        self.pfProc.pfComp.Compute_current()      # Topological charge current
        # pfProc.pfComp.Compute_alpha_tilde()  # Dislocation density tensor

        # -----------------------------
        # Mechanics update
        # -----------------------------

        self.mec_proc.mecFE.Q.x.array[:] = self.ProcEXT.Compute_Q(amps)  # Configurational distortion
        self.mec_proc.Get_Curls()        # Compute curls
        self.mec_proc.mecFE.alpha.interpolate(self.mec_proc.mecComp.curlQ)
        if self.simparams.SOLVE_MEC :
            self.mec_proc.solveUPperp() # gived chi_p or up_perp

            self.mec_proc.mecFE.J.interpolate(self.pfProc.pfComp.J)
            self.mec_proc.solve_Up_para() # gived chi_p or up_perp
            self.mec_proc.combine_UP()
            self.mec_proc.solveU()           # Solve mechanical equilibrium
            self.mec_proc.extract_UE()       # Extract elastic distortion

            self.mec_proc.compute_sym()      # Compute symmetric parts of UE and Q
            self.mec_proc.Get_Stress()       # Calculate stresses
            self.mec_proc.Get_Curls()        # Compute curls
            self.mec_proc.mecFE.alpha.interpolate(self.mec_proc.mecComp.curlQ)