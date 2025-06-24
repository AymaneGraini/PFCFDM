"""
    This is the main class the handles all the phase field part of the simulation.

    This is the facade the user instanciate and interacts with. All other classes are varaibles within this class.
"""

import numpy as np
import dolfinx
import basix
import dolfinx.fem as fem
import ufl
import scipy.ndimage as ndimage
from .PfFe import *
from .PFC4th import *
from .PfComp import *
from .PfIO import *
# from .PFC6th import * #TODO add PFC6
from .PFSolver import *
from PFCproc_TODO.InitialC import * # TODO this should not be here ugly

class PfProc:
    def __init__(self,
                 domain:dolfinx.mesh.Mesh,
                 pfc_params: PfcParams,
                 sim_params: SimParams,
                 file : dolfinx.io.XDMFFile
                 ):
        """
            Initializes the PfProc class, the main class that handles all the phase field part of the simulation.

            Args:
                domain (dolfinx.mesh.Mesh): The computational domain for the phase field simulation.
                pfc_params (PfcParams): Parameters specific to the phase field model.
                sim_params (SimParams): Simulation parameters.
                file (dolfinx.io.XDMFFile): The XDMF file to which output will be written.
        """

        self.pfc_params=pfc_params
        self.sim_params=sim_params
    
        #Initializes the phase field Finite Element class of the desired degree
        if pfc_params.deg == 4:
            self.pfFe = PFC4(domain,pfc_params,sim_params)
        elif pfc_params.deg == 6:
            raise ValueError("Maybe i forgot to updathe the H-1 interface")
            self.pfFe = PFC6(domain,pfc_params,sim_params)
        else : 
            raise ValueError("Phase Field model not supported")
        
        # Initializes the phase field solver, the phase field computation Class and the input/output class
        self.pfSolver     = PFSolver(self.pfFe) #
        self.pfComp      = PfComp(self.pfFe)
        self.pfio      = PfIO(self.pfFe,self.pfComp,file)

        #Defines arrays to store the average and energy history
        self.avg_history=[]
        self.E_history=[]

    def Initialize_crystal(self, defects):
        """
        Initializes the phase field with a crystal structure based on the provided defects array.
        
        This method calls `initialize_from_burgers` to set the initial phase field configuration based on the defects in the crystal structure.
        which returns a function to be intropolated into the phase field variable. psi0

        Then we compute the average of the initial phase field configuration and store it in `avg_history`. 

        And finally, it interpolates the initial phase field configuration into and `psiout` for output

        Args:
            defects (np.ndarray): An array representing the defects in the crystal structure, of shape (N,3) where N is the number of defects and each defect is represented by its coordinates (x, y) and its burgers vector.

        """
        Amp =  lambda avg,r : (1/5)*(np.absolute(avg)+(1/3)*np.sqrt(15*r-36*avg**2))
        A= Amp(self.pfc_params.avg,self.pfc_params.r)
        self.pfFe.psi0.interpolate(lambda x: initialize_from_burgers(self.pfc_params.qs,self.pfc_params.ps,defects,A,self.pfc_params.avg)(x))


        self.pfFe.psiout.interpolate(self.pfFe.psi0)
        avg1= fem.assemble_scalar(fem.form(self.pfFe.psiout*self.pfFe.dx))/(self.sim_params.L*self.sim_params.H)
        print("THe average after initializing is ", avg1)
        self.avg_history.append(avg1)

    def Intialize_random(self,seed):
        """
        Initializes the phase field with a random Noisy configuration with average equal to `pfc_params.avg`.
        A random number generator is used to create a random phase field configuration, which is then interpolated into the `psi0` variable.

        Args:
            seed (int): The seed for the random number generator to ensure reproducibility.
        """ 
        rng = np.random.default_rng(seed)
        initialCpsi = lambda x : (rng.random(x.shape[1])-0.5)+self.pfc_params.avg
        self.pfFe.psi0.interpolate(initialCpsi)
        self.pfFe.psiout.interpolate(self.pfFe.psi0)
        avg1= fem.assemble_scalar(fem.form(self.pfFe.psiout*self.pfFe.dx))/(self.sim_params.L*self.sim_params.H)
        print("THe average in initing is ", avg1)
        self.avg_history.append(avg1)

    def Initialize(self,f0):
        """
            INitializes the phase field with a given function `f0`, f0 should be a lambda function of compatible shape with the mesh.
        """
        self.pfFe.psi0.interpolate(f0)
        self.pfFe.psiout.interpolate(self.pfFe.psi0)

    
    def init_solver(self):
        """
            Initializes the phase field solver by creating the main and auxiliary forms needed for the phase field equations.
    
        """
        self.pfFe.create_main_forms()
        self.pfFe.create_auxiliary_forms()


    def Configure_solver(self):
        """
            Configures the phase field solver by assembly the matrix, and creating a KSP solver with the desired tolerances, precondtioners, and monitors and other parameters.
        """
        self.pfSolver.configure_solver()
        self.pfSolver.set_chi_solver()

    def Solve(self):
        """
            Solves the phase field equations using the configured solver with a target average equal to the initial value.
            It updates the average history and interpolates the solution into `psiout`.
        """
        self.pfSolver.solve(self.avg_history[0])


    def get_SH_Energy(self):
        """
        Computes the total energy of the phase field system by assembling the energy form Into a scalar and returning its value.
        """
        E = fem.assemble_scalar(self.pfFe.Energyform)
        self.E_history.append(E)
        return E
    
    def get_chi(self):
        """
        Returns the current value of the chi variable, which represents the weak laplacian of the phase field.
        """
        self.pfSolver.get_chi()

    def Correct(self):
        """
            Applies an average correction step if relevant, then intepolates the obtained fied into the previous solution variables and the current solution into psiout for output.
        """
        self.pfFe.correct()

    def write_output(self,t: float) -> None:
        """
        Writes the output to the XDMF file at time `t`.

        Args:
            t (float): The current time step for which the output is written.
        """
        self.pfio.write_output(t)