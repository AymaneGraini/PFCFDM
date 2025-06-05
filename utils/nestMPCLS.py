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
import dolfinx_mpc

class mpcLinearSolverNest(dfxmpc.LinearProblem):
    def __init__(self,
                a, L,
                mpcs, #this is a list
                bcs                   = None,
                u                     = None,
                petsc_options         = None,
                AddNullSpace          = None,
                form_compiler_options = None,
                jit_options           = None):
        # super().__init__(a, L, mpc, bcs = None, u = None, petsc_options = None, form_compiler_options = None, jit_options = None)
        
        form_compiler_options = {} if form_compiler_options is None else form_compiler_options
        jit_options = {} if jit_options is None else jit_options
        self._a = _fem.form(a, jit_options=jit_options, form_compiler_options=form_compiler_options)
        self._L = _fem.form(L, jit_options=jit_options, form_compiler_options=form_compiler_options)

        for _mpc in mpcs:
            if not _mpc.finalized:
                raise RuntimeError("The multi point constraint has to be finalized before calling initializer")
            
        self._mpcs = mpcs

        self._A = dolfinx_mpc.create_matrix_nest(a, mpcs)


        self._b = dolfinx_mpc.create_vector_nest(L, mpcs)
 

        self._x =  self._b.copy()

        self.bcs = [] if bcs is None else bcs

        self._solver = PETSc.KSP().create(self._mpcs[0].function_space.mesh.comm)  # type: ignore
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
        dolfinx_mpc.assemble_matrix_nest(self._A, self.a, self._mpcs, self.bcs)
        self._A.assemble()
        assert self._A.assembled

        
    def solve(self) -> _fem.Function:
        """Solve the problem.

        Returns:
            petsc nestVec containing the solution"""
        for bsub in  self._b .getNestSubVecs():    
            with bsub.localForm() as b_loc:
                b_loc.set(0)

        dolfinx_mpc.assemble_vector_nest(self._b, self._L, self._mpcs)
        dolfinx.fem.petsc.apply_lifting_nest(self._b, self._a, self.bcs)

        self._solver.solve(self._b, self._x)


        """
            The solution requires back substitution and it's done in the solver after calling .solve()
        """
        return self._x
