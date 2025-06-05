import dolfinx_mpc as dfxmpc
import dolfinx.fem.petsc
import ufl
from dolfinx import fem as _fem
from dolfinx_mpc.assemble_matrix import assemble_matrix, create_sparsity_pattern
from dolfinx_mpc.assemble_vector import apply_lifting, assemble_vector
from petsc4py import PETSc
from dolfinx import cpp as _cpp
from dolfinx import fem as _fem
from dolfinx import la as _la

class mpcLinearSolver(dfxmpc.LinearProblem):
    def __init__(self, a, L, mpc, bcs = None, u = None, petsc_options = None,AddNullSpace=None, form_compiler_options = None, jit_options = None):
        # super().__init__(a, L, mpc, bcs = None, u = None, petsc_options = None, form_compiler_options = None, jit_options = None)
        
        form_compiler_options = {} if form_compiler_options is None else form_compiler_options
        jit_options = {} if jit_options is None else jit_options
        self._a = _fem.form(a, jit_options=jit_options, form_compiler_options=form_compiler_options)
        self._L = _fem.form(L, jit_options=jit_options, form_compiler_options=form_compiler_options)

        if not mpc.finalized:
            raise RuntimeError("The multi point constraint has to be finalized before calling initializer")
        self._mpc = mpc
        # Create function containing solution vector
        if u is None:
            self.u = _fem.Function(self._mpc.function_space)
        else:
            if u.function_space == self._mpc.function_space:
                self.u = u
            else:
                raise ValueError(
                    "The input function has to be in the function space in the multi-point constraint",
                    "i.e. u = dolfinx.fem.Function(mpc.function_space)",
                )
        self._x = self.u.x.petsc_vec

        # Create MPC matrix
        pattern = create_sparsity_pattern(self._a, self._mpc)
        pattern.finalize()
        self._A = _cpp.la.petsc.create_matrix(self._mpc.function_space.mesh.comm, pattern)

        self._b = _la.create_petsc_vector(
            self._mpc.function_space.dofmap.index_map, self._mpc.function_space.dofmap.index_map_bs
        )
        self.bcs = [] if bcs is None else bcs

        self._solver = PETSc.KSP().create(self.u.function_space.mesh.comm)  # type: ignore
        self._solver.setOperators(self._A)

        # Give PETSc solver options a unique prefix
        solver_prefix = "dolfinx_mpc_solve_{}".format(id(self))
        self._solver.setOptionsPrefix(solver_prefix)

        # Set PETSc options
        opts = PETSc.Options()  # type: ignore
        opts.prefixPush(solver_prefix)
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
        self.PC = self._solver.getPC()
        self.PC.setReusePreconditioner(True) #Note This the only difference
        opts.prefixPop()
        self._solver.setFromOptions()

    def assembleBiLinear(self):
        self._A.zeroEntries()
        assemble_matrix(self._a, self._mpc, bcs=self.bcs, A=self._A)
        self._A.assemble()
        assert self._A.assembled

        
    def solve(self) -> _fem.Function:
        """Solve the problem.

        Returns:
            Function containing the solution"""

        with self._b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(self._L, self._mpc, b=self._b)
        # Apply boundary conditions to the rhs
        apply_lifting(self._b, [self._a], [self.bcs], self._mpc)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
        _fem.petsc.set_bc(self._b, self.bcs)

        
        # Solve linear system and update ghost values in the solution
        # print("SOlving with ", self._x.array_r)
        self._solver.solve(self._b, self._x)
        self.u.x.scatter_forward()
        self._mpc.backsubstitution(self.u)

        return self.u
