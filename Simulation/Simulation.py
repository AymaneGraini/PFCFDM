
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
    def __init__(self,pfcparms,simparams,geometry,mechparams):

        self.pfcparms = pfcparms
        self.simparams = simparams
        self.geometry = geometry
        self.mechparams = mechparams

        self.average_history=[]
        self.errors_history=[]
        self.rel_erros_history=[]
        self.avg_history=[]
        self.mechanical_dissipation=[]
        self.t = 0

    def initialize():
        pass
    

    def Compute_scalars(self):
        self.mechanical_dissipation.append(fem.assemble_scalar(dissipation))
        component_errors = [error_L2(self.mec_proc.mecFE.UEsym.sub(i), self.mec_proc.mecComp.Qsym.sub(i)) for i in range(4)]
        self.erros_history.append(component_errors)
        self.rel_erros_history.append([error_L2_rel(self.mec_proc.mecFE.UEsym.sub(i), self.mec_proc.mecComp.Qsym.sub(i)) for i in range(4)])
        self.avg_history.append(fem.assemble_scalar(self.pfProc.pfFe.Avg_form))

    def run():
        pass
    
    def write_output():
        pass
        