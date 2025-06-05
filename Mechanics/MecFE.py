import dolfinx
import dolfinx.fem as fem
import ufl
import basix
from utils.MPCLS import *
from utils.pbcs import *
from utils.MyAlgebra import *
from utils.monitor import *
from utils.utils import *
from Simulation.Parameters import*

class MecFE :
    """
        A class used to handle the main FE part of the Mechanics problem,
        Handles main FE field to solve the mechancis.
        It doesn't contain auxiliary fields 
    """
    def __init__(self,                
                 domain:dolfinx.mesh.Mesh,
                 mech_params: MechParams,
                 sim_params: SimParams
                 ):
        
        self.periodic_UP = mech_params.periodic_UP
        self.periodic_u  = mech_params.periodic_u
        self.domain      = domain
        self.dx          = ufl.Measure("dx",domain=domain)
        self.ds          = ufl.Measure("ds",domain=domain)
        self.sim_params  = sim_params
        self.mech_params = mech_params
        self.set_spaces()
        self.pbcs_UPperp = PeriodicBC_geometrical(domain, self.vector_sp4,1,[]) if self.periodic_UP else None
        self.pbcs_u      = PeriodicBC_geometrical(domain, self.vector_sp2_quad,1,[]) if self.periodic_u else None
        self.set_funcs()
        self.f    = fem.Constant(domain,mech_params.f) #BOdy foyrce

    def set_spaces(self):
        self.elem            = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1)
        self.Ts_P3           = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(3,3))
        self.Vs_P3           = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(3,))
        self.Ts_P2           = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(2,2))
        self.Vs_P4           = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(4,))
        self.Vs_P2           = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(2,))
        self.Vs_P2_quad      = basix.ufl.element("Lagrange", self.domain.basix_cell(), 2, shape=(2,))

        self.outsp           = fem.functionspace(self.domain, self.elem)
        self.tensor_sp3      = fem.functionspace(self.domain, self.Ts_P3)
        self.tensor_sp2      = fem.functionspace(self.domain, self.Ts_P2)
        self.vector_sp3      = fem.functionspace(self.domain, self.Vs_P3)
        self.vector_sp2      = fem.functionspace(self.domain, self.Vs_P2)
        self.vector_sp2_quad = fem.functionspace(self.domain, self.Vs_P2_quad)
        self.vector_sp4      = fem.functionspace(self.domain, self.Vs_P4)
    
    def set_funcs(self):
        self.UPperp = fem.Function(self.tensor_sp2,name="UPperp")
        self.UPpara = fem.Function(self.tensor_sp2,name="UPpara")
        self.UP     = fem.Function(self.tensor_sp2,name="UP")
        self.U      = fem.Function(self.tensor_sp2,name="U")
        self.Q      = fem.Function(self.tensor_sp2,name="Q")
        self.UE     = fem.Function(self.tensor_sp2,name="UE")
        self.UEsym  = fem.Function(self.tensor_sp2,name="UEsym")
        self.u_disp = fem.Function(self.vector_sp2_quad)
        self.u_out  = fem.Function(self.vector_sp2,name="u")

        if self.periodic_UP:
            self.Uperp3 = fem.Function(self.pbcs_UPperp.function_space,name="UPperp3")
        else:
            self.Uperp3 = fem.Function(self.tensor_sp3,name="UPperp3")

        self.U4          = fem.Function(self.vector_sp4,name="UP4")
        self.alpha       = fem.Function(self.tensor_sp3,name="alpha")

        # Define test and trial functions
        if self.periodic_UP :
            self.u_inc = ufl.TrialFunction(self.pbcs_UPperp.function_space)
            self.v_inc = ufl.TestFunction(self.pbcs_UPperp.function_space)
        else:
            self.u_inc = ufl.TrialFunction(self.tensor_sp3)
            self.v_inc = ufl.TestFunction(self.tensor_sp3)

        if self.periodic_u :
            self.u_e = ufl.TrialFunction(self.pbcs_u.function_space)
            self.v_e = ufl.TestFunction(self.pbcs_u.function_space)
        else:
            self.u_e = ufl.TrialFunction(self.vector_sp2_quad)
            self.v_e = ufl.TestFunction(self.vector_sp2_quad)