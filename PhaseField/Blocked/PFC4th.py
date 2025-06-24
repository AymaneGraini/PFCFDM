"""
    A class that implements the unconserved dynamic evolution equation: Swift-Hohenberg model 
    This class formulates the coupled problem as a blocked problem and not a mixed one.
"""

import numpy as np
import dolfinx
import basix
import dolfinx.fem as fem
import ufl
import scipy.ndimage as ndimage
from .PfFe import *



class PFC4(PFFe):
    """
    A class that implements the unconserved dynamic evolution equation: Swift-Hohenberg model.
    
    It innherits from PFFe and sets up the additional spaces, functions, and forms for the specific to this model.
    
    The common spaces, functions, and forms are set up in the PFFe class.

    """
    def __init__(self,
                 domain:dolfinx.mesh.Mesh,
                 pfc_params: PfcParams,
                 sim_params: SimParams
                 )->None:
        """
        Initializes the PFC4 class with the given domain, phase field parameters, and simulation parameters.


        Args:
            domain (dolfinx.mesh.Mesh): The computational domain.
            pfc_params (PfcParams): Phase field parameters including wavevectors, average, and other model parameters.
            sim_params (SimParams): Simulation parameters including time step, maximum time, and coefficients for the evolution equation.
        """
        # Initialize the parent class with the domain, phase field parameters, and simulation parameters
        super().__init__(domain,pfc_params,sim_params)
        
        self.ConservationMethod = pfc_params.ConservationMethod #Conservation method for the average, either "LM" or "scalar" or "field"
        self.set_spaces()#Defines the functional spaces,
        self.set_funcs()#Defines the functions and trial/test functions,

        self.set_projection_basis() #Sets the projection basis in order to compute the average as a dot product with the basis

        self.pbcs = [PeriodicBC_geometrical_nest(domain, self.main_space,1,[])  for _ in range(2)] if self.pfc_params.periodic else None #TODO Does it work in a blocked monolithic problem 

        #Define the maps and sizes for the blocked problem
        if self.ConservationMethod == "LM":
            self.maps = [(self.main_space.dofmap.index_map, self.main_space.dofmap.index_map_bs),
                         (self.main_space.dofmap.index_map, self.main_space.dofmap.index_map_bs),
                         (self.real_space.dofmap.index_map, self.real_space.dofmap.index_map_bs)]
        else:
            self.maps = [(self.main_space.dofmap.index_map, self.main_space.dofmap.index_map_bs),
                         (self.main_space.dofmap.index_map, self.main_space.dofmap.index_map_bs)]
    
    
        #Define the sizes of the maps
        self.sizes = [imap.size_local * bs for imap, bs in self.maps]

        #Define the offsets for the map in order to access the local dofs of each function in the blocked problem
        self.offsets=[0]
        for size in self.sizes[:-1]:
            self.offsets.append(self.offsets[-1] + size)

    def set_spaces(self):
        """
        Sets the functional spaces for the phase field model.
        This defines the main space used an then calls the parent class method to set the common spaces.
        """
        # Define the main space for the scalar fields
        self.elem = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1)
        self.main_space = fem.functionspace(self.domain, self.elem)
        # Calls the mother class method to set the common spaces
        super().set_spaces()

    def set_funcs(self):
        """
        Sets the functions and trial/test functions for the phase field model.

        The main are:

        - psi0 : previous value of psi at time :math:`t`
        - chi0 : previous value of chi at time :math:`t`
        - psi_sol : current value of psi at time :math:`t+dt`
        - chi_sol : current value of chi at time :math:`t+dt`
        - lmbda : Lagrange multiplier for the average conservation

        The Trial functions are:

        - psi_current : trial function for psi at time :math:`t+dt`
        - chi_current : trial function for chi at time :math:`t+dt`
        - _lm : trial function for the Lagrange multiplier at time :math:`t+dt`

        The Test functions are:

        - q : test function for psi
        - v : test function for chi 
        - dl : test function for the Lagrange multiplier
        """


        super().set_funcs()
        self.psi0 = fem.Function(self.main_space) # previous psi
        self.chi0 = fem.Function(self.main_space,name="chi")  # previous chi
        self.lmbda = fem.Function(self.real_space)  #Lagrange multiplier

        self.psi_sol = fem.Function(self.main_space) #Solved for (current sol)
        self.chi_sol = fem.Function(self.main_space) #Solved for (current sol)
        
        # Define the trial functions for the main space and real space
        (self.psi_current, self.chi_current) =  ufl.TrialFunction(self.main_space), ufl.TrialFunction(self.main_space)
        self._lm = ufl.TrialFunction(self.real_space)

        # Define the test functions for the main space and real space
        self.q, self.v = ufl.TestFunction(self.main_space), ufl.TestFunction(self.main_space)
        self.dl = ufl.TestFunction(self.real_space)


    def create_auxiliary_forms(self):
        """
            Creates the following auxiliary forms:
            - To compute the weak laplacan form a given psi by solving a linear problem
            - To compute the swift hohenberg energy
            - To compute the average of psi.

            The linear form to obtain the weak laplacian of psi0 (:math:`\\psi^t`) is of the form:

            .. math::

                L_t(v) = -\int_{\\Omega} \\nabla \\psi^{t} \cdot \\nabla v \, d\\Omega

            The energy form is of the form:

            .. math::

                E(\\psi^t, \\chi^t) = \\int_{\\Omega}  \\frac{1}{2}(1-r)(\\psi^{t})^2 + \\frac{1}{4}(\\psi^{t})^4 + \\frac{1}{2}(\\chi^{t})^2 - ||\\psi^{t}||^2 \, d\\Omega

            Finally, the average of psi is computed as:

            .. math::

                \\tilde{\\psi^t} = \\frac{1}{L \\times H} \\int_{\\Omega} \\psi^{t} \, d\\Omega
        """
        r          = self.pfc_params.r

        self.L_chi      = fem.form(-1.0*ufl.inner(ufl.grad(self.psi0),ufl.grad(self.v))*self.dx) # form to extract the weak laplacian of psi0

        #Form to compute the Swift hohenberg energy given psi0 and chi0
        self.Energyform = fem.form(((1/2)*(1-r)*self.psi0**2+(1/4)*self.psi0**4+
                                    (1/2)*self.chi0**2-ufl.inner(ufl.grad(self.psi0),ufl.grad(self.psi0)))
                                    *self.dx) 
        
        #Form to compute the average of psi0
        self.Avg_form = fem.form((1/(self.sim_params.L*self.sim_params.H))*self.psi0*self.dx)
    

    def create_main_forms(self):
        """
        Creates the forms for the PFC4 model.
        Creates the bilinear and linear form depending on the conservation method and the simulation parameters.
        
        This function also compiles the forms using dolfinx FFCX form compiler.

        Generally the bilinear form is of the form:


        .. math::

            \\begin{aligned}
            a(\\psi^{t+dt}, \\chi^{t+dt},\\lambda^{t+dt}; q, v, l) = & (\\frac{1}{dt C_{sh}}+1-r)\int_{\\Omega} \\psi^{t+dt} q  \, d\\Omega \\\\
                & -\int_{\\Omega} \\nabla \chi^{t+dt} \cdot \\nabla v + 2 \chi q + \lambda^{t+dt} q\, d\\Omega+ \int_{\\Omega} \\nabla \\psi^{t+dt}  \cdot \\nabla v \, d\\Omega \\\\
                    & +\int_{\\Omega} \chi^{t+dt} v \, d\\Omega + \int_{\\Omega} \\psi^{t+dt} l \, d\\Omega
            \\end{aligned}

        Whereas the linear form is of the form:

        .. math::
            L( q, v, l) = \\frac{1}{dt C_{sh}} \int_{\\Omega} \\psi^{t} q \, d\\Omega- \\int_{\\Omega} (\\psi^{t})^3 q \, d\\Omega - \\frac{C_w}{C_{sh}} \int_{\\Omega} \\left(\\frac{\delta \mathcal{F}_u}{\delta \psi} \\right)_t q \, d\\Omega  + \int_{\\Omega}  \\tilde{h} l \, d\\Omega

        where :math:`\\tilde{h}` is the average of psi over the domain. This cannot be computed at compilation time, so instead it is computed as 0 in order, only as a place holder then the value is changed during assembly, prior to solving the problem. Like it was done in scifem dolfinx example.

        The form are assembled in a blocked manner,

        .. math::
            a = \\begin{bmatrix}
            a_{\psi \psi} & a_{\psi \\chi} & a_{\psi \lambda} \\\\
            a_{\chi \psi} & a_{\chi \chi} & 0 \\\\
            a_{\lambda \psi} & 0 & 0
            \\end{bmatrix} \quad \\text{and} \quad 
            L = \\begin{bmatrix}
            L_{\psi} \\\\
            0 \\\\
            \\tilde{h}
            \end{bmatrix}
    NOTE:
        The form cannot handle the case where :math:`C_w=0`, so it is handled separately within an if statement.

        In the case where no Lagrange multiplier is used, the last term is not included in the bilinear form and the linear form is simplified accordingly.
        The matrix form is simplfied to the  :math:`2 \\times 2` upper left matrix, and the last entry of the vector :math:`L` is not included.

        """
        r          = self.pfc_params.r
        dt         = self.sim_params.dt
        Csh        = self.sim_params.Csh
        Cw         = self.sim_params.Cw


        self.a11    = (1/(dt*Csh)+1-r)*ufl.inner(self.psi_current,self.q)*self.dx
        self.a12    = -1.0*ufl.inner(ufl.grad(self.chi_current),ufl.grad(self.q))*self.dx+2*ufl.inner(self.chi_current,self.q)*self.dx
        self.a13    = 1.0*ufl.inner(self._lm,self.q)*self.dx

        self.a21    = ufl.inner(ufl.grad(self.psi_current),ufl.grad(self.v))*self.dx
        self.a22    = ufl.inner(self.chi_current,self.v)*self.dx
        self.a23    = None

        self.a31    = ufl.inner(self.psi_current,self.dl)*self.dx
        self.a32    = None
        self.a33    = None

        if self.ConservationMethod == "LM":
            self.a_pfc  = fem.form([[self.a11,self.a12,self.a13],
                        [self.a21,self.a22,self.a23],
                        [self.a31,self.a32,self.a33]])
        else:
            self.a_pfc  = fem.form([[self.a11,self.a12],
                        [self.a21,self.a22]])

        if Cw==0:
            print("Iiniting with Cw=0")
            if self.ConservationMethod == "LM":
                self.L_pfc = fem.form([ ufl.inner((1/(dt*Csh))*self.psi0-self.psi0**3,self.q)*self.dx,
                                    ufl.inner(fem.Constant(self.domain, PETSc.ScalarType(0)), self.v) * self.dx,
                                    ufl.inner(fem.Constant(self.domain, PETSc.ScalarType(0)), self.dl) * self.dx
                                    ]) 
            else: 
                self.L_pfc = fem.form([ ufl.inner((1/(dt*Csh))*self.psi0-self.psi0**3,self.q)*self.dx,
                                    ufl.inner(fem.Constant(self.domain, PETSc.ScalarType(0)), self.v) * self.dx
                                    ])
                                   
        else:
            if self.ConservationMethod == "LM":
                self.L_pfc = fem.form([ ufl.inner((1/(dt*Csh))*self.psi0-self.psi0**3-(Cw/Csh)*self.dFQW,self.q)*self.dx,
                                        ufl.inner(fem.Constant(self.domain, PETSc.ScalarType(0)), self.v) * self.dx,
                                    ufl.inner(fem.Constant(self.domain, PETSc.ScalarType(0)), self.dl) * self.dx
                                        ])
            else:
                self.L_pfc = fem.form([ ufl.inner((1/(dt*Csh))*self.psi0-self.psi0**3-(Cw/Csh)*self.dFQW,self.q)*self.dx,
                                        ufl.inner(fem.Constant(self.domain, PETSc.ScalarType(0)), self.v) * self.dx
                                        ])


    def correct(self):
        """
            This function is used to perform three main things :

            - Correct the average of psiout to be equal to the desired average given in the parameters using the chosen method.
            - Interpolate the current solution of psi into psiout for output
            - Interpolate the current solution (psi,chi) into the previous values (psi0, chi0) for the next time step.
        """
        if self.ConservationMethod=="LM":
                self.psiout.interpolate(self.psi_sol)
                self.psi0.interpolate(self.psi_sol)

        elif self.ConservationMethod=="scalar":
            self.psiout.interpolate(self.psi_sol)
            psi_avg = fem.assemble_scalar(fem.form(self.psiout*ufl.dx))/(self.sim_params.L*self.sim_params.H)
            self.psiout.x.array[:] = self.psiout.x.array +self.pfc_params.avg - psi_avg
            self.psi0.interpolate(self.psiout)

        elif self.ConservationMethod=="field":
            # TODO maybe chi must be corrected as well?
            self.psiout.interpolate(self.psi_sol)
            beta = (self.psiout.x.petsc_vec.dot(self.b_basis.x.petsc_vec)-self.pfc_params.avg)/self.b_basis_norm
            self.psiout.x.petsc_vec.axpy(-1.0*beta, self.b_basis.x.petsc_vec)
            self.psi0.interpolate(self.psiout)


        self.chi0.interpolate(self.chi_sol)

        self.psi0.x.scatter_forward()
        self.chi0.x.scatter_forward()

