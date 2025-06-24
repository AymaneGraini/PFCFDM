"""
This class implements small deformations elasticity.

This module defines the MecFE class, which handles the finite element
part of a mechanics problem, including the definition of function spaces,
trial and test functions, and periodic boundary conditions if applicable.

This class is part of a larger simulation framework and relies on the
dolfinx library for finite element methods and ufl for defining
variational forms. It also uses basix for defining finite element
elements and function spaces.
"""

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

        Args:
            domain (dolfinx.mesh.Mesh): The mesh domain for the simulation.
            mech_params (MechParams): Mechanical parameters.
            sim_params (SimParams): Simulation parameters.
    """
    def __init__(self,                
                 domain:dolfinx.mesh.Mesh,
                 mech_params: MechParams,
                 sim_params: SimParams
                 ):
        
        self.periodic_UP = mech_params.periodic_UP # boolean for periodic boundary conditions on plastic distortion
        self.periodic_u  = mech_params.periodic_u # boolean for periodic boundary conditions on displacement
        self.domain      = domain # Mesh domain for the simulation
        self.dx          = ufl.Measure("dx",domain=domain) # Integration measure over the domain
        self.ds          = ufl.Measure("ds",domain=domain) # Integration measure over the boundary
        self.sim_params  = sim_params # dataclass containing simulation parameters
        self.mech_params = mech_params #dataclass containing mechanical parameters
        self.set_spaces() # Define the finite element spaces for the main fields
        self.pbcs_UPperp = PeriodicBC_geometrical(domain, self.vector_sp4,1,[]) if self.periodic_UP else None # Define periodic boundary conditions for the plastic distortion :math:`\\mathbf{U_p}`.
        self.pbcs_u      = PeriodicBC_geometrical(domain, self.vector_sp2_quad,1,[]) if self.periodic_u else None # Define periodic boundary conditions for the displacement :math:`\\mathbf{u}`.
        self.set_funcs() # Define all the FE functions for each relevant field using the FE spaces stored in mecFE
        self.f    = fem.Constant(domain,mech_params.f) # Body force applied to the system, defined as a constant function over the domain.

    def set_spaces(self):
        """
        Define the finite element spaces for the main fields.

        Starts with defining the basix elements for the different function spaces,
        then creates the corresponding function spaces using `dolfinx.fem.functionspace`.

        Function spaces defined:
            - **outsp**: Scalar function space for the main field.
            - **tensor_sp3**: Tensor function space for 3x3 tensors.
            - **tensor_sp2**: Tensor function space for 2x2 tensors.
            - **vector_sp3**: Vector function space for 3d vectors.
            - **vector_sp2**: Vector function space for 2D vectors.
            - **vector_sp2_quad**: 2D vector function space with quadratic elements.
            - **vector_sp4**: 4D vector space used for periodic boundary conditions 

        Notes:
            The `vector_sp4` space is used for periodic boundary conditions for the plastic distortion 
            :math:`\\mathbf{U_p}`, since `dolfinx.mpc` cannot handle periodic boundary conditions directly 
            for tensors. Therefore, a vector space with 4 components is used to represent the tensor 
            in a way that can be handled by the MPC (Multi-point constraints) framework. 
            Utility functions are then used to convert between vector and tensor representations.
            """
       
        # Define the basix elements for the different function spaces
        self.elem            = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1)
        self.Ts_P3           = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(3,3))
        self.Vs_P3           = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(3,))
        self.Ts_P2           = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(2,2))
        self.Vs_P4           = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(4,))
        self.Vs_P2           = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(2,))
        self.Vs_P2_quad      = basix.ufl.element("Lagrange", self.domain.basix_cell(), 2, shape=(2,))

        # Create the function spaces using dolfinx.fem.functionspace
        self.outsp           = fem.functionspace(self.domain, self.elem)
        self.tensor_sp3      = fem.functionspace(self.domain, self.Ts_P3)
        self.tensor_sp2      = fem.functionspace(self.domain, self.Ts_P2)
        self.vector_sp3      = fem.functionspace(self.domain, self.Vs_P3)
        self.vector_sp2      = fem.functionspace(self.domain, self.Vs_P2)
        self.vector_sp2_quad = fem.functionspace(self.domain, self.Vs_P2_quad)
        self.vector_sp4      = fem.functionspace(self.domain, self.Vs_P4)
    
    def set_funcs(self):
        """
            Define all the FE functions for each relevant field using the FE spaces stored in mecFE.
            These functions represent the main fields in the mechanics problem, including:

                - **UPperp**: Incompatible part of the plastic distortion tensor.
                - **UPpara**: Compatible part of the plastic distortion tensor. 
                - **UP**: Total plastic distortion tensor. :math:`\\mathbf{U_p} =\\mathbf{U_p}^\\perp +\\mathbf{U_p}^\\parallel`.
                - **U**: Total distortion tensor. :math:`\\mathbf{U} = \\mathbf{U_p} + \\mathbf{U_e}`.
                - **Q**: Configurational distortion tensor coming from phase field crystal.
                - **UE**: Elastic part of the deformation gradient tensor.
                - **UEsym**: Symmetric part of the elastic deformation gradient tensor.
                - **u_disp**: Displacement vector field, interpolated using the quadratic space.
                - **u_out**: Output displacement vector field, a projectio of **u_disp** onto the linear space, because xmdf files do not support quadratic elements. 
                - **Uperp3**: Perpendicular part of the plastic distortion tensor in a 3D space, used for periodic boundary conditions.
                - **U4**: Vector field for the plastic distortion tensor in a 4D space, used for periodic boundary conditions.
                - **alpha**: Tensor field for the alpha parameter in a 3D space, used for periodic boundary conditions.

            Defines trial and test functions for the variational formulation:
                - **u_inc**: Trial function for the plastic distortion tensor, defined in the periodic space if applicable.
                - **v_inc**: Test function for the plastic distortion tensor, defined in the periodic space if applicable.
                - **u_e**: Trial function for the displacement vector field, defined in the periodic space if applicable.
                - **v_e**: Test function for the displacement vector field, defined in the periodic space if applicable.
        

        """

        # Define the FE functions for each relevant field using the FE spaces stored in mecFE
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