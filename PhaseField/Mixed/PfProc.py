import numpy as np
import dolfinx
import basix
import dolfinx.fem as fem
import ufl
import scipy.ndimage as ndimage
from .PfFe import *
from .PFC4th import *
from .PFC6th import *
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
        
        self.pfc_params=pfc_params
        self.sim_params=sim_params
    
        if pfc_params.deg == 4:
            self.pfFe = PFC4(domain,pfc_params,sim_params)
        elif pfc_params.deg == 6:
            # raise ValueError("Maybe i forgot to updathe the H-1 interface")
            self.pfFe = PFC6(domain,pfc_params,sim_params)
        else : 
            raise ValueError("Phase Field model not supported")
        
        self.pfSolver     = PFSolver(self.pfFe)
        self.pfComp      = PfComp(self.pfFe)
        self.pfio      = PfIO(self.pfFe,self.pfComp,file)
        self.avg_history=[]
        self.E_history=[]
        
    def Initialize_crystal(self, defects):
        Amp =  lambda avg,r : (1/5)*(np.absolute(avg)+(1/3)*np.sqrt(15*r-36*avg**2))
        A= Amp(self.pfc_params.avg,self.pfc_params.r)
        self.pfFe.zeta0.sub(0).interpolate(lambda x: initialize_from_burgers(self.pfc_params.qs,self.pfc_params.ps,defects,A,self.pfc_params.avg)(x))
        self.pfFe.psiout.interpolate(self.pfFe.zeta0.sub(0))
        avg1= fem.assemble_scalar(fem.form(self.pfFe.psiout*self.pfFe.dx))/(self.sim_params.L*self.sim_params.H)
        self.avg_history.append(avg1)

    
    def init_solver(self):
        self.pfFe.create_forms()

    def Configure_solver(self):
         self.pfSolver.configure_solver()

    def Solve(self):
        self.pfSolver.solve()
        self.pfFe.psiout.interpolate(self.pfFe.SH_sol.sub(0))

    def get_SH_Energy(self):
        E = fem.assemble_scalar(self.pfFe.Energyform)
        self.E_history.append(E)
        return E
    def Correct(self):
        self.pfFe.correct()

    def write_output(self,t: float) -> None:
        self.pfio.write_output(t)