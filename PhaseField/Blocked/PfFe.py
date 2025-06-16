import numpy as np
import dolfinx
import basix
import dolfinx.fem as fem
import ufl
import ufl.form
import ufl.measure
from utils.nestMPCLS import *
from utils.pbcs import *
from utils.MyAlgebra import *
from utils.monitor import *
from utils.utils import *
import scipy.ndimage as ndimage
import jax.numpy as jnp
from Simulation.Parameters import*
import scifem



class PFFe:
    def __init__(self,
                 domain:dolfinx.mesh.Mesh,
                 pfc_params: PfcParams,
                 sim_params: SimParams
                 )->None:
        """
            A class used to handle the main FE part of the Phasefield problem ,
            Handles main FE field to solve the PhaseField.
            It doesn't contain auxiliary fields such as the current, the amplitudes, distortion

            This is an abstract class "Interface" that needs to be implemented
            depending on the model
        """
                
        self.domain     = domain
        self.pfc_params = pfc_params
        self.sim_params = sim_params
        self.dx         = ufl.Measure("dx",domain=domain)

    def set_spaces(self):
        self.scalar_sp  = fem.functionspace(self.domain,self.elem)
        self.Ts_P3      = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1,shape=(3,3))
        self.tensor_sp3 = fem.functionspace(self.domain, self.Ts_P3)

        self.Ts_P2      = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1,shape=(2,2))
        self.tensor_sp2 = fem.functionspace(self.domain, self.Ts_P2)

        self.Vs_P2      = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1,shape=(2,))
        self.vector_sp2 = fem.functionspace(self.domain, self.Vs_P2)
        self.real_space = scifem.create_real_functionspace(self.domain)

    def set_funcs(self):
        """
            Defines the shared functions between all models, trial, split and tests functions remain
            specific to each model and their definition is implemented in each model, separetly
        """        

        self.psiout  = fem.Function(self.scalar_sp,name="Psi") #function to store the output psi = mu.sub(0)
        self.dFQW    = fem.Function(self.scalar_sp,name="dFQW")
        self.corr    = fem.Function(self.scalar_sp,name="Correction")

        self.Re_amps = []
        self.Im_amps = []
        for i,q in enumerate(self.pfc_params.qs):
            self.Re_amps.append(fem.Function(self.scalar_sp,name="RealAmp"+str(i),dtype=np.float64))
            self.Im_amps.append(fem.Function(self.scalar_sp,name="ImagAmp"+str(i),dtype=np.float64))

        self.Re_amps_old=[]
        self.Im_amps_old=[]
        for i,q in enumerate(self.pfc_params.qs):
            self.Re_amps_old.append(fem.Function(self.scalar_sp,name="RealAmp_old"+str(i),dtype=np.float64))
            self.Im_amps_old.append(fem.Function(self.scalar_sp,name="ImagAmp_old"+str(i),dtype=np.float64))

    def set_projection_basis(self):
        self.b_basis = fem.Function(self.scalar_sp,name="basis")
        self.phi_i_intg = fem.assemble_vector(fem.form((1/(self.sim_params.L*self.sim_params.H))*ufl.TestFunction(self.scalar_sp) * ufl.dx))
        self.b_basis.x.array[:] = self.phi_i_intg.array.copy()
        self.b_basis_norm = np.dot(self.phi_i_intg.array, self.phi_i_intg.array)
   