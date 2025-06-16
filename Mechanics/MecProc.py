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
        part of the simulation
    """
    def __init__(self,
                 domain     : dolfinx.mesh.Mesh,
                 mech_params: MechParams,
                 sim_params : SimParams,
                 file : dolfinx.io.XDMFFile
                 ): 
        
        self.sim_params = sim_params

        #Arrays to store main indicators 
        self.FUQ        = []
        self.avgs       = []
        self.L2divs     = []
        self.L2divsQ    = []
        self.FSH        = []

        #Dirichlet Bcs for Upperp and u
        self.mecFE      = MecFE(domain,mech_params,sim_params)
        self.mecComp      = MecComp(self.mecFE)
        self.mecio      = MecIO(file,self.mecFE,self.mecComp)



    def init_solver(self,
                    bcsUperp: typing.List[_fem.DirichletBC],
                    bcsu    : typing.List[_fem.DirichletBC]): 
        self.mecSolver  = MecSolver(self.mecFE)
        self.bcsUperp=bcsUperp
        self.bcs_u= bcsu

    def ConfigureSolver_UPperp(self):
        self.mecSolver.configure_solver_UPperp(self.bcsUperp)

    def ConfigureSolver_u(self):
        self.mecSolver.configure_solver_U(self.bcs_u)

    def solveUPperp(self):
        self.mecSolver.solve_UPperp(self.bcsUperp)

    def solveU(self):
        self.mecSolver.solve_U(self.bcs_u)


    def update_UP(self,pfc_solver):
        """
            Update UP using a forward Euler scheme : 
                Up(t+dt) = Up(t) + dt*J #TODO + or - ?
            with J = (alpha x vd) is the plastic rate but also 
            the current linked with the conservation of burgers vector
            Thus the Argument is the whole PFCsolver that contains J.
        """
        self.mecFE.UP.x.array[:]-=self.sim_params.dt*pfc_solver.pfComp.J.x.array[:]  # TODO maybe petsc.axpy is much faster ...

    def combine_UP(self):
        """
            Combines compatibale and incompatble parts of Up
                Up = Up∥ + Up⟂
        """
        self.mecFE.UP.x.array[:] = self.mecFE.UPpara.x.array+self.mecFE.UPperp.x.array
    

    def extract_UE(self):
        """
            Extracts the elastic part of the distortion : Ue=U-Up
        """
        self.mecFE.UE.x.array[:] = self.mecFE.U.x.array-self.mecFE.UP.x.array

    def Get_Stress(self):
        self.mecComp.compute_stresses()

    def Get_Divergence(self):
        self.mecComp.compute_divergence()
    def Get_Curls(self):
        self.mecComp.compute_curls()
        
    def compute_sym(self):
        self.mecComp.compute_sym()

    def write_output(self,t:float):
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