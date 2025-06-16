import dolfinx
import dolfinx.fem as fem
import ufl
import basix
from utils.MPCLS import *
from utils.pbcs import *
from utils.MyAlgebra import *
from utils.monitor import *
from utils.utils import *
from .MecFE import *

class MecSolver :
    """
        A class used to handle the FEM Solver of the Mechanics problem
    """
    def __init__(self, mecFe : MecFE):
        self.mecFe = mecFe
        self.create_forms_UPperp()
        self.create_forms_U()

    def create_forms_UPperp(self):
        nv= ufl.FacetNormal(self.mecFe.domain)
        if self.mecFe.periodic_UP:
            self.mecFe.a_inc = fem.form(ufl.inner(tcurl2d(self.mecFe.u_inc,"v"),tcurl2d(self.mecFe.v_inc,"v"))*self.mecFe.dx
                                        +ufl.inner(ufl.div(v2T(self.mecFe.u_inc,2)),ufl.div(v2T(self.mecFe.v_inc,2)))*self.mecFe.dx
                                        +self.mecFe.mech_params.Cx * ufl.inner(ufl.dot(v2T(self.mecFe.u_inc,2), nv), 
                                                                               ufl.dot(v2T(self.mecFe.v_inc,2), nv)) * self.mecFe.ds)
            
            self.mecFe.L_inc = fem.form(ufl.inner(-self.mecFe.alpha,tcurl(extendT(v2T(self.mecFe.v_inc,2))))*self.mecFe.dx)
        else :
            self.mecFe.a_inc = fem.form(ufl.inner(tcurl(self.mecFe.u_inc),tcurl(self.mecFe.v_inc))*self.mecFe.dx
                                        +ufl.inner(tdiv(self.mecFe.u_inc),tdiv(self.mecFe.v_inc))*self.mecFe.dx
                                        + self.mecFe.mech_params.Cx * ufl.inner(vdot(self.mecFe.u_inc, nv),
                                                                                vdot(self.mecFe.v_inc, nv)) * self.mecFe.ds)
            
            self.mecFe.L_inc = fem.form(ufl.inner(-self.mecFe.alpha,tcurl(self.mecFe.v_inc))*self.mecFe.dx)

    def create_forms_U(self):
        lambda_ = self.mecFe.mech_params.lambda_
        mu_     = self.mecFe.mech_params.mu
        Cw      = self.mecFe.sim_params.Cw
        Cel     = self.mecFe.mech_params.Cel

        if self.mecFe.sim_params.penalty_u:
            self.mecFe.a_u = fem.form(ufl.inner(Cel*sigma(epsilon(self.mecFe.u_e),lambda_,mu_)
                                                +Cw*epsilon(self.mecFe.u_e), epsilon(self.mecFe.v_e )) * self.mecFe.dx )
            
            self.mecFe.L_u = fem.form(ufl.inner(Cel*sigma(ufl.sym(self.mecFe.UP),lambda_,mu_)
                                                +Cw*(ufl.sym(self.mecFe.UP)+ufl.sym(self.mecFe.Q)), epsilon(self.mecFe.v_e )) * self.mecFe.dx
                                                +ufl.inner(self.mecFe.f,self.mecFe.v_e)* self.mecFe.dx)
        else:
            self.mecFe.a_u = fem.form(ufl.inner(Cel*sigma(epsilon(self.mecFe.u_e),lambda_,mu_), ufl.grad(self.mecFe.v_e )) * self.mecFe.dx )
            
            self.mecFe.L_u = fem.form(ufl.inner(Cel*sigma(ufl.sym(self.mecFe.UP),lambda_,mu_), ufl.grad(self.mecFe.v_e )) * self.mecFe.dx
                                      +ufl.inner(self.mecFe.f,self.mecFe.v_e)* self.mecFe.dx)
    

    def configure_solver_UPperp(self,bcs)->None:
        if self.mecFe.periodic_UP :
            opts={
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol":1e-8,
            "ksp_atol":1e-10,
            "ksp_reuse_preconditioner": True,
            "ksp_initial_guess_nonzero":True
            }
            self.mecFe.problem_UPperp = mpcLinearSolver(
                    self.mecFe.a_inc,
                    self.mecFe.L_inc,
                    self.mecFe.pbcs_UPperp,
                    bcs=bcs,
                    petsc_options=opts
                )
            print("solving Uperp with", self.mecFe.problem_UPperp._solver.getPC().getType(),self.mecFe.problem_UPperp._solver.getType())
            self.mecFe.problem_UPperp.assembleBiLinear()
        else:
            self.mecFe.A_inc = fem.petsc.assemble_matrix(self.mecFe.a_inc, bcs=bcs)
            self.mecFe.A_inc.assemble()
            self.mecFe.problem_UPperp = PETSc.KSP().create(self.mecFe.domain.comm)
            self.mecFe.problem_UPperp.setOperators(self.mecFe.A_inc)
            self.mecFe.problem_UPperp.setInitialGuessNonzero(True)
            self.mecFe.problem_UPperp.setType(PETSc.KSP.Type.BCGS)
            self.mecFe.problem_UPperp.rtol = 1e-8
            self.mecFe.problem_UPperp.atol = 1e-10
            # self.mecFe.uPperpmoni = CVmonitor()
            # self.mecFe.problem_UPperp.setMonitor(self.mecFe.uPperpmoni.monitor)
            pc = self.mecFe.problem_UPperp.getPC()
            pc.setType(PETSc.PC.Type.GAMG)
            pc.setReusePreconditioner(True)
            # print("inited with ", self.mecFe.problem_UPperp.getPC().getType(),self.mecFe.problem_UPperp.getType())
            # print("Symmetric:", self.mecFe.A_inc.isSymmetric(tol=1e-8))


    def configure_solver_U(self,bcs)->None:
        if self.mecFe.periodic_u :
            optsu={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_rtol":1e-8,
            "ksp_atol":1e-10,
            "pc_factor_mat_solver_type": "superlu_dist", #superlu_dist
            "ksp_reuse_preconditioner": True
            }
            self.mecFe.problem_U = mpcLinearSolver(
                    self.mecFe.a_u,
                    self.mecFe.L_u,
                    self.mecFe.pbcs_u,
                    bcs=bcs,
                    petsc_options=optsu
                )
            print("solving U with", self.mecFe.problem_U._solver.getPC().getType(),self.mecFe.problem_U._solver.getType())
            self.mecFe.problem_U.assembleBiLinear()
        else:
            self.mecFe.A_u = fem.petsc.assemble_matrix(self.mecFe.a_u, bcs=bcs)
            self.mecFe.A_u.assemble()
            if self.mecFe.mech_params.addNullspace:
                ns= buildnullspace(self.mecFe.domain, self.mecFe.vector_sp2_quad)
                # self.mecFe.A_u.setOption(petsc4py.PETSc.Mat.Option.SYMMETRIC, True)
                # self.mecFe.A_u.setOption(petsc4py.PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
                assert ns.test(self.mecFe.A_u)
                print("Here is the nullspace, ", ns.view())
                self.mecFe.A_u.setNullSpace(ns)
            self.mecFe.problem_U = PETSc.KSP().create(self.mecFe.domain.comm)
            self.mecFe.problem_U.setOperators(self.mecFe.A_u)
            self.mecFe.problem_U.setType(PETSc.KSP.Type.PREONLY)
            self.mecFe.problem_U.rtol = 1e-8
            self.mecFe.problem_U.atol = 1e-10
            # self.mecFe.Umoni = CVmonitor()
            # self.mecFe.problem_U.setMonitor(self.mecFe.Umoni.monitor)
            pc = self.mecFe.problem_U.getPC()
            pc.setType(PETSc.PC.Type.LU)
            pc.setFactorSolverType(petsc4py.PETSc.Mat.SolverType.MUMPS)
            pc.setReusePreconditioner(True)
            self.mecFe.b_u = fem.petsc.create_vector(self.mecFe.L_u)
        
    def solve_UPperp(self,bcs):
        if self.mecFe.periodic_UP:
            with dolfinx.common.Timer() as t_cpu:
                sol = self.mecFe.problem_UPperp.solve()
                print("Uperp done in : %s" % t_cpu.elapsed()[0])
                self.mecFe.U4.interpolate(fem.Expression(sol,self.mecFe.vector_sp4.element.interpolation_points()))
                self.mecFe.UPperp.interpolate(fem.Expression(v2T(self.mecFe.U4,2),self.mecFe.tensor_sp2.element.interpolation_points()))
            # self.mecFe.uPperpmoni.plot()
            # self.mecFe.uPperpmoni.n+=1
        else:
            with dolfinx.common.Timer() as t_cpu:
                b_inc = fem.petsc.create_vector(self.mecFe.L_inc)
                with b_inc.localForm() as loc_b:
                    loc_b.set(0)
                fem.petsc.assemble_vector(b_inc, self.mecFe.L_inc)
                if len(bcs)>0:
                    fem.petsc.apply_lifting(b_inc, [self.mecFe.a_inc], bcs=[bcs])
                    b_inc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    for bc in bcs:
                        bc.set(b_inc.array_w)
                else:
                   b_inc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                self.mecFe.problem_UPperp.solve(b_inc, self.mecFe.Uperp3.x.petsc_vec)
                print("UPperp done in : %s" % t_cpu.elapsed()[0])
                self.mecFe.UPperp.interpolate(fem.Expression(restrictT(self.mecFe.Uperp3),self.mecFe.tensor_sp2.element.interpolation_points()))

    def solve_U(self,bcs):
        if self.mecFe.periodic_u:
            with dolfinx.common.Timer() as t_cpu:
                soli = self.mecFe.problem_U.solve()
                print("U done in : %s" % t_cpu.elapsed()[0])
                self.mecFe.U.interpolate(fem.Expression(ufl.grad(soli),self.mecFe.tensor_sp2.element.interpolation_points()))
                self.mecFe.u_out.interpolate(fem.Expression(soli,self.mecFe.vector_sp2.element.interpolation_points()))
        else:
            with dolfinx.common.Timer() as t_cpu:
                with self.mecFe.b_u.localForm() as loc_b:
                    loc_b.set(0)
                fem.petsc.assemble_vector(self.mecFe.b_u, self.mecFe.L_u)
                if len(bcs)>0:
                    fem.petsc.apply_lifting(self.mecFe.b_u, [self.mecFe.a_u], bcs=[bcs])
                    self.mecFe.b_u.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    for bc in bcs:
                        bc.set(self.mecFe.b_u.array_w)
                else:
                   self.mecFe.b_u.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                if self.mecFe.mech_params.addNullspace:
                    self.mecFe.A_u.getNullSpace().remove(self.mecFe.b_u)
                self.mecFe.u_disp.x.petsc_vec.zeroEntries()
                # PETSc.Log.reset()
                # PETSc.Log.begin()
                self.mecFe.problem_U.solve(self.mecFe.b_u, self.mecFe.u_disp.x.petsc_vec)
                # PETSc.Log.view()
                print("U done in : %s" % t_cpu.elapsed()[0])
                self.mecFe.U.interpolate(fem.Expression(ufl.grad(self.mecFe.u_disp),self.mecFe.tensor_sp2.element.interpolation_points()))
                self.mecFe.u_out.interpolate(fem.Expression(self.mecFe.u_disp,self.mecFe.vector_sp2.element.interpolation_points()))
            # self.mecFe.Umoni.export()
            # self.mecFe.Umoni.n+=1
