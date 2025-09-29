"""
    MecProc is the main Facade for the user to interact with the mechancis
    part of the simulation .

    The user can use this class to set up the mechanics problem,
    configure the solver, solve the mechanics equations, and write output.

"""

from Mechanics.MecFE import *
from Mechanics.MecSolver import *
from Mechanics.MecComp import *
from Mechanics.MecIO import *
from Simulation.Parameters import *
from dolfinx import fem as _fem
import typing


class MecProc:
    """
        MecProc is the main Facade for the user to interact with the mechancis
        part of the simulation.
        The user can use this class to set up the mechanics problem,
        configure the solver, solve the mechanics equations, and write output.

        Args:
            domain (dolfinx.mesh.Mesh): The mesh domain for the simulation.
            mech_params (MechParams): data class for Mechanical parameters.
            sim_params (SimParams):  data class for Simulation parameters.
            file (dolfinx.io.XDMFFile): The XDMF file to write output to.
    """
    def __init__(self,
                 domain     : dolfinx.mesh.Mesh,
                 mech_params: MechParams,
                 sim_params : SimParams,
                 file : dolfinx.io.XDMFFile
                 ): 
        """
        Initialize the MecProc class.
        This method sets up the mechanics problem by initializing the finite element spaces and function spaces, and prepares the output file for writing results.
        It also initializes arrays to store main indicators for the simulation, such as the L2 norms of divergence, average values, and energies.

        Args:
            domain (dolfinx.mesh.Mesh): Domain mesh for the simulation.
            mech_params (MechParams): Data class for mechanical parameters.
            sim_params (SimParams): Data class for simulation parameters.
            file (dolfinx.io.XDMFFile): XDMF file to write output to.
        """
        self.sim_params = sim_params

        #Arrays to store main indicators 
        self.FUQ        = []
        self.avgs       = []
        self.L2divs     = []
        self.L2divsQ    = []
        self.FSH        = []

        self.mecFE      = MecFE(domain,mech_params,sim_params) #Define the finite element part of the mechanics problem.
        self.mecComp      = MecComp(self.mecFE) #define the mechanics computation part, containing auxiliary fields and computations.
        self.mecio      = MecIO(file,self.mecFE,self.mecComp) # define the mechanics IO part, used to write output in XDMF format.



    def init_solver(self,
                    bcsUperp: typing.List[_fem.DirichletBC],
                    bcsu    : typing.List[_fem.DirichletBC],
                    Nmn_Bcs
                    ): 
        """
        Initialize the MecSolver with the given boundary conditions.
        This method sets up the MecSolver with the provided boundary conditions for the plastic distortion and displacement fields.

        Args:
            bcsUperp (typing.List[_fem.DirichletBC]): Dirchlet boundary conditions for the plastic distortion field.
            bcsu (typing.List[_fem.DirichletBC]): Dirichlet boundary conditions for the displacement field.
        """
        self.mecSolver  = MecSolver(self.mecFE,Nmn_Bcs) # Initialize the MecSolver with the finite element part of the mechanics problem.
        self.bcsUperp=bcsUperp # Set the boundary conditions for the plastic distortion field.
        self.bcs_u= bcsu # Set the boundary conditions for the displacement field.
        
    def ConfigureSolver_UPperp(self):
        """
        Configure the MecSolver for the plastic distortion field.
        This method sets up the solver for the plastic distortion field with the specified boundary conditions.

        It sets the type of the Krylov subspace solver,  the precondtionner used, the tolerance, and the maximum number of iterations.

        It also assembles the bilinear form using the corresponding function spaces and boundary conditions.
        """
        self.mecSolver.configure_solver_UPperp(self.bcsUperp)

    def ConfigureSolver_u(self):
        """
            Configure the MecSolver for the elastic distortion field, the main uknown is the displacement field.
            This method sets up the solver for the displacement vector field with the specified boundary conditions.

            It sets the type of the Krylov subspace solver,  the precondtionner used, the tolerance, and the maximum number of iterations.

            It also assembles the bilinear form using the corresponding function spaces and boundary conditions.
        """
        self.mecSolver.configure_solver_U(self.bcs_u)


    def Configure_solver_zp(self):
        self.mecSolver.configure_solver_ZP()
    
    def solve_Up_para(self):
        self.mecSolver.solve_zp()
        self.mecFE.UPpara.interpolate(fem.Expression(ufl.grad(self.mecFE.zp_new),self.mecFE.tensor_sp2.element.interpolation_points()))
        self.mecFE.zp_old.interpolate(self.mecFE.zp_new)

    def init_UP_para(self,defects,h):
        exp = ufl.as_ufl(0)
        x,y = ufl.SpatialCoordinate(self.mecFE.domain)
        for defect in defects:
            xi,yi,bi = defect
            exp+=bi[0]*(1+ufl.tanh(-1*(x-xi)/h))*ufl.exp(-(y-yi)**2/h**2)
        exp*=(1/(2*np.sqrt(np.pi)*h))
        self.mecFE.UPpara.sub(1).interpolate(fem.Expression(exp,self.mecFE.tensor_sp2.sub(1).element.interpolation_points())) 

    def solveUPperp(self):
        """
            Calls the  preivously built Petsc solver to solve the div-curl system defining the plastic distortion field.
            it also assembles the linear form into the RHS of the system using the bcs.
        """
        self.mecSolver.solve_UPperp(self.bcsUperp)

    def solveU(self):
        """
            Calls the preivously built Petsc solver to solve the elasticty problem defined by the displacement vector field.
            it also assembles the linear form into the RHS of the system using the bcs.
        """
        self.mecSolver.solve_U(self.bcs_u)


    def update_UP(self,pfc_solver=None): #TODO add type hinting it maybe needs import.
        """
            Update UP using a forward Euler scheme : 
                :math:`\mathbf{Up}(t+dt) = \mathbf{Up}(t) + dt\mathcal{J}`
            with J = (alpha x vd) is the plastic rate but also 
            the current linked with the conservation of burgers vector
            Thus the Argument is the whole PFCsolver that contains :math:`\mathcal{J}`.

            Args:
                pfc_solver (PFCSolver): The PFC solver instance that contains the current plastic rate J.
        """
        dt=self.mecFE.sim_params.dt
        if pfc_solver:
            self.mecFE.UP.x.array[:]-=self.sim_params.dt*pfc_solver.pfComp.J.x.array[:]  # TODO maybe petsc.axpy is much faster ...
        else:
            raise NotImplementedError
            # self.mecFE.UP.interpolate(
            #     fem.Expression(ufl.grad(self.mecFE.zp_new)+self.mecFE.UPperp,self.mecFE.tensor_sp2.element.interpolation_points()))    
            # self.mecFE.zp_old.interpolate(self.mecFE.zp_new)

            # self.mecComp.UPdot.interpolate(fem.Expression(restrictT(tcrossv(self.mecFE.alpha,self.mecComp.V_pk)),self.mecFE.tensor_sp2.element.interpolation_points()))
            # self.mecFE.UP.x.array[:]+=self.sim_params.dt*self.mecComp.UPdot.x.array[:]

        
    def combine_UP(self):
        """
            Combines compatibale and incompatble parts of :math:`\mathbf{Up}`
                :math:`\mathbf{Up} = \mathbf{Up}^\parallel + \mathbf{Up}^\perp`
        """
        self.mecFE.UP.x.array[:] = self.mecFE.UPpara.x.array+self.mecFE.UPperp.x.array
    

    def extract_UE(self):
        """
            Extracts the elastic part of the distortion :math:`\mathbf{Ue} = \mathbf{U} - \mathbf{Up}`
        """
        self.mecFE.UE.x.array[:] = self.mecFE.U.x.array-self.mecFE.UP.x.array

    def Get_Stress(self):
        """
            Compute the stress due to :math:`\\mathbf{Q}` as :math:`\\mathbb{C}:sym\\mathbf{Q}`
            and the elastic stress either:
                un-Coupled: :math:`\\mathbb{C}:sym(\\mathbf{U_e})`
                or Coupled : :math:`\\mathbb{C}:sym(\\mathbf{U_e}) + C_w \\, sym(\\mathbf{U_e}-\\mathbf{Q})`
            Results are stored in self.mecComp.sigmaUe and self.mecComp.sigmaQ
        """
        self.mecComp.compute_stresses()

    def Get_Divergence(self):
        """
            Compute the divergence of both stresses  :math:`\\mathbf{div}(\sigma_{U_e})` and :math:`\\mathbf{div}(\sigma_Q)`
            coming from self.compute_stresses() 

            Results are stored in self.mecComp.divsUe and self.mecComp.divsQ
        """
        self.mecComp.compute_divergence()

    def Get_Curls(self):
        """
            Compute the curl of the fields :math:`\\nabla \\times \\mathbf{U_e}`,
            :math:`\\nabla \\times \\mathbf{U_p}` and :math:`\\nabla \\times \\mathbf{Q}`
        """
        self.mecComp.compute_curls()
    
    def PK_velocity(self):
        """Compute the velocity field :math:`\\mathbf{v_d}` using the PK velocity definition in the coupled case.
        """
        self.mecComp.compute_velocity()

    def compute_sym(self):
        """
            computes the symmetric part of the fields
            :math:`sym(\\mathbf{U_e})` and :math:`sym(\\mathbf{Q})`
            Results are stored in self.mecFE.UEsym and self.mecFE.Qsym
        """
        self.mecComp.compute_sym()

    def write_output(self,t:float):
        """Writes the output for the current time step.

        Args:
            t (float): The current time.    
        """
        self.mecio.write_output(t)

    def compute_indicators(self,pfcSolver):
        #TODO this should not be in here, it should be in a class of simulation 
        # and it must eat pfcsolver and mecsolver
        """
            Compute the main indicators to follow and
            rate the simulation:
                - L2 norm of div_SigmaUe and div_SigmaQ
                - L2 norm of Ue-Q
                - Swift-hohenberg energy
                - Average of psi

        Args:
            pfcSolver (PFCSolver): The PFC solver instance that contains the psi field
        """
        L2div= np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(self.mecFE.divsUe,self.mecFE.divsUe)*self.dx)))
        L2divQ= np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(self.mecFE.divsQ,self.mecFE.divsQ)*self.dx)))
        avg= fem.assemble_scalar(fem.form(pfcSolver.psiout*ufl.dx))/( self.sim_params.L* self.sim_params.H)

        error = error_L2(self.mecFE.UE, self.mecFE.Q, degree_raise=3)
        E     = fem.assemble_scalar(fem.form(((1/2)*(-pfcSolver.pfc_params['r']+1)*pfcSolver.psi0**2
                                          +(1/4)*pfcSolver.psi0**4+(1/2)*(pfcSolver.chi0)**2
                                          -ufl.inner(ufl.grad(pfcSolver.psi0),ufl.grad(pfcSolver.psi0)))
                                          *self.dx))
        self.FUQ.append(error)
        self.FSH.append(E)
        self.L2divs.append(L2div)
        self.L2divsQ.append(L2divQ)
        self.avgs.append(avg)