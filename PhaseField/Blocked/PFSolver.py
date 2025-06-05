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
from dolfinx.la import create_petsc_vector_wrap
from .PfFe import *
import time

class PFSolver:
    def __init__(self,pfFe:PFFe):
        self.pfFe = pfFe

    def configure_solver(self)->None:
        if self.pfFe.pfc_params.periodic:
            opts={
            "ksp_type": "preonly",
            "pc_type": "lu",
            # "pc_factor_mat_solver_type": "superlu_dist",
            "ksp_reuse_preconditioner": True
                }
            self.problem_pfc = mpcLinearSolverNest(
                    self.pfFe.a_pfc,
                    self.pfFe.L_pfc,
                    self.pfFe.pbcs,
                    bcs=[],
                    petsc_options=opts
                )
            self.problem_pfc.assembleBiLinear()
        else:
            self.A_pfc = fem.petsc.assemble_matrix_nest(self.pfFe.a_pfc, bcs=[])
            self.A_pfc.assemble()
            self.problem_pfc = PETSc.KSP().create(self.pfFe.domain.comm)
            # self.psimonitor = CVmonitor()
            # self.problem_pfc.setMonitor(self.psimonitor.monitor)
            self.problem_pfc.setOperators(self.A_pfc)
            # self.problem_pfc.setInitialGuessNonzero(True)
            self.problem_pfc.setType(PETSc.KSP.Type.PREONLY)

            pc = self.problem_pfc.getPC()
            pc.setType(PETSc.PC.Type.LU)
            # pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
            # pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.UPPER)

            # pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.A11)  # Or USER if you're customizing

            size = self.A_pfc.getSize()[0] // 2
            # is0 = PETSc.IS().createStride(size, 0, 1, comm=PETSc.COMM_WORLD)
            # is1 = PETSc.IS().createStride(size, size, 1, comm=PETSc.COMM_WORLD)
            # pc.setFieldSplitIS(('psi', is0), ('chi', is1))

            # self.problem_pfc.setUp()

            # kspA, kspS = pc.getFieldSplitSubKSP()

            # kspA.setType('preonly')
            # kspA.getPC().setType('cholesky')  

            # kspS.setType('preonly')
            # kspS.getPC().setType('hypre')  

            pc.setReusePreconditioner(True)
            self.x_pfc = PETSc.Vec().createNest([create_petsc_vector_wrap(self.pfFe.psi_sol.x), create_petsc_vector_wrap(self.pfFe.chi_sol.x)])
            self.x_pfc.zeroEntries()
            
    def solve(self):
        if self.pfFe.pfc_params.periodic:
            # with dolfinx.common.Timer() as t_cpu:
                sol = self.problem_pfc.solve()
                
                self.pfFe.psi_sol.x.petsc_vec.setArray(sol.getNestSubVecs()[0])
                self.pfFe.chi_sol.x.petsc_vec.setArray(sol.getNestSubVecs()[1])

                self.pfFe.pbcs[0].backsubstitution(self.pfFe.psi_sol)
                self.pfFe.pbcs[1].backsubstitution(self.pfFe.chi_sol)

        else:
                self.b_pfc = fem.petsc.assemble_vector_nest(self.pfFe.L_pfc)
                # print("SOlving with ", self.x_pfc.getArray())
                self.problem_pfc.solve(self.b_pfc, self.x_pfc)
                
                # offset =self.pfFe.main_space.dofmap.index_map.size_local * self.pfFe.main_space.dofmap.index_map_bs
                # u.x.array[:offset] = self.x.array_r[:offset]
                # p.x.array[: (len(self.x.array_r) - offset)] = self.pfFe.main_space.x.array_r[offset:]
    
    def set_chi_solver(self):
        self.chimonitor = CVmonitor()
        self.A_chi = fem.petsc.assemble_matrix(fem.form(self.pfFe.a22), bcs=[])
        self.A_chi.assemble()
        self.problem_chi = PETSc.KSP().create(self.pfFe.domain.comm)
        self.problem_chi.setMonitor(self.chimonitor.monitor)
        self.problem_chi.setOperators(self.A_chi)
        self.problem_chi.setType(PETSc.KSP.Type.PREONLY)
        pc = self.problem_chi.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setReusePreconditioner(True)


    def get_chi(self):
        self.b_chi = fem.petsc.assemble_vector(self.pfFe.L_chi)
        t1=time.time()
        self.problem_chi.solve(self.b_chi, self.pfFe.chi0.x.petsc_vec)
        t2=time.time()
        print("Chi solver in ", t2-t1)


