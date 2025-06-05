import numpy as np
import dolfinx
import basix
import dolfinx.fem as fem
import ufl
import ufl.form
from utils.MPCLS import *
from utils.pbcs import *
from utils.MyAlgebra import *
from utils.monitor import *
from utils.utils import *
import scipy.ndimage as ndimage
import jax.numpy as jnp

from .PfFe import *


class PFSolver:
    def __init__(self,pfFe:PFFe):
        self.pfFe = pfFe

    def configure_solver(self)->None:
        if self.pfFe.pfc_params.periodic:
            print("Periodic Bcs")
            opts={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "superlu_dist",
            "ksp_reuse_preconditioner": True
                }
            self.problem_pfc = mpcLinearSolver(
                    self.pfFe.a_pfc,
                    self.pfFe.L_pfc,
                    self.pfFe.pbcs,
                    bcs=[],
                    petsc_options=opts
                )
            self.problem_pfc.assembleBiLinear()
        else:
            self.A_pfc = fem.petsc.assemble_matrix(self.pfFe.a_pfc, bcs=[])
            self.A_pfc.assemble()
            self.problem_pfc = PETSc.KSP().create(self.pfFe.domain.comm)
            self.problem_pfc.setOperators(self.A_pfc)
            self.problem_pfc.setType(PETSc.KSP.Type.PREONLY)
            pc = self.problem_pfc.getPC()
            pc.setType(PETSc.PC.Type.LU)
            pc.setReusePreconditioner(True)

    def solve(self):
        if self.pfFe.pfc_params.periodic:
            with dolfinx.common.Timer() as t_cpu:
                self.pfFe.SH_sol = self.problem_pfc.solve()
        else:
            with dolfinx.common.Timer() as t_cpu:
                b_pfc = fem.petsc.create_vector(self.pfFe.L_pfc)
                with b_pfc.localForm() as loc_b:
                    loc_b.set(0)
                fem.petsc.assemble_vector(b_pfc, self.pfFe.L_pfc)
                self.problem_pfc.solve(b_pfc, self.pfFe.SH_sol.x.petsc_vec)