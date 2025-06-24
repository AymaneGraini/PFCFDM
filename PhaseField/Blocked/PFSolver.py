"""
A class to solve the phase field equations using a linear solver, it handles the assembly of the system, the solution of the equations, and the management of boundary conditions.

"""

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
from dolfinx.cpp.la.petsc import scatter_local_vectors, get_local_vectors

class PFSolver:
    def __init__(self,pfFe:PFFe):
        """
        Initializes the PFSolver class.
        Args:
            pfFe (PFFe): An instance of the PFFe class containing phase field equations, functional spaces and functions.
        """
        self.pfFe = pfFe

    def configure_solver(self)->None:
        """
        Configures the linear solver for the phase field equations.
        This method sets up the solver based on whether the problem is periodic or not.
        It also initializes the necessary matrices and vectors for the solver.

        It does the assembly of the matrix, so that it is stored in the solver object and can be re-used in the next time step.

        The methos also initializes a KSP solver from the PETSc library, sets the type of the solver, and configures the preconditioner used.

        """
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
            # Defines a block matrix using the compiled bilinear form.
            self.A_pfc = fem.petsc.assemble_matrix_block(self.pfFe.a_pfc, bcs=[])
            self.A_pfc.assemble() # assembles the matrix, so that it is stored in the solver object and can be re-used whenever ksp.solver() is called.

            # Initializes A KSP solver
            self.problem_pfc = PETSc.KSP().create(self.pfFe.domain.comm)
            # self.psimonitor = CVmonitor()
            # self.problem_pfc.setMonitor(self.psimonitor.monitor)
            self.problem_pfc.setOperators(self.A_pfc) # Sets the matrix for the solver.
            # self.problem_pfc.setInitialGuessNonzero(True)
            self.problem_pfc.setType(PETSc.KSP.Type.PREONLY) #Sets the solver type

            pc = self.problem_pfc.getPC() # Gets the preconditioner from the solver.
            pc.setType(PETSc.PC.Type.LU)  # Sets the preconditioner type to LU.
            pc.setFactorSolverType("mumps") # Sets the factor solver type to MUMPS


            # pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
            # pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.UPPER)

            # pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.A11)  # Or USER if you're customizing

            # size = self.A_pfc.getSize()[0] // 2
            # is0 = PETSc.IS().createStride(size, 0, 1, comm=PETSc.COMM_WORLD)
            # is1 = PETSc.IS().createStride(size, size, 1, comm=PETSc.COMM_WORLD)
            # pc.setFieldSplitIS(('psi', is0), ('chi', is1))

            # self.problem_pfc.setUp()

            # kspA, kspS = pc.getFieldSplitSubKSP()

            # kspA.setType('preonly')
            # kspA.getPC().setType('cholesky')  

            # kspS.setType('preonly')
            # kspS.getPC().setType('hypre')  

            pc.setReusePreconditioner(True) # Sets the preconditioner to be reused in subsequent solves without the need to reassemble it since the matrix does not change and both are stored in the solver object.

            
            # creates a zero vector to store the solution
            self.x_pfc = self.A_pfc.createVecLeft() 
            self.x_pfc.zeroEntries()

    def solve(self,avg):
        """
        Solves the phase field equations using the configured solver and updates the solution vectors.

        Args:
            avg (float): The target value to be used in the assembly of the right-hand side vector when a lagrange multiplier is used
        """
        if self.pfFe.pfc_params.periodic:
            # with dolfinx.common.Timer() as t_cpu:
                sol = self.problem_pfc.solve()
                
                self.pfFe.psi_sol.x.petsc_vec.setArray(sol.getNestSubVecs()[0])
                self.pfFe.chi_sol.x.petsc_vec.setArray(sol.getNestSubVecs()[1])

                self.pfFe.pbcs[0].backsubstitution(self.pfFe.psi_sol)
                self.pfFe.pbcs[1].backsubstitution(self.pfFe.chi_sol)

        else:
                # Assembles the right-hand side vector for the phase field equations.
                self.b_pfc = fem.petsc.assemble_vector_block(self.pfFe.L_pfc,self.pfFe.a_pfc,[])

                if self.pfFe.pfc_params.ConservationMethod=="LM":
                    # Access the local vectors from the PETSc vector to change the value of the corresponding dofs that correspond to the average value of the phase field.
                    b_local = get_local_vectors(self.b_pfc, self.pfFe.maps)

                    b_local[2][:] = avg*self.pfFe.sim_params.H*self.pfFe.sim_params.L

                    scatter_local_vectors(
                            self.b_pfc,
                            b_local,
                            self.pfFe.maps,
                        )
                # UPdate ghost values. TODO this is irrelevant now.
                self.b_pfc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

                # Solves the proble with RHS set to b_pfc and solution vector set to x_pfc.
                self.problem_pfc.solve(self.b_pfc, self.x_pfc)

                # Updates the solution vectors with the values from the local vector.
                self.pfFe.psi_sol.x.array[:self.pfFe.sizes[0]] = self.x_pfc.array_r[self.pfFe.offsets[0]:self.pfFe.offsets[1]]
                self.pfFe.chi_sol.x.array[:self.pfFe.sizes[1]] = self.x_pfc.array_r[self.pfFe.offsets[1]:self.pfFe.offsets[2]]


    def set_chi_solver(self):
        """
            Defines a linear problem to extract the weak laplacian of a given psi
        """
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
        """
            Calls the previously defined linear problem to extract the weak laplacian of a given psi.
        """
        self.b_chi = fem.petsc.assemble_vector(self.pfFe.L_chi)
        t1=time.time()
        self.problem_chi.solve(self.b_chi, self.pfFe.chi0.x.petsc_vec)
        t2=time.time()
        print("Chi solver in ", t2-t1)


