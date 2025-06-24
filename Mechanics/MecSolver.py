"""
This module contains the MecSolver class, which is responsible for solving the mechanics problem using finite element methods.
It handles the creation and compilation of forms for the incompatible plastic distortion and displacement fields, configures the solver, and provides methods for solving the system.
It handles also the assembly of the bilinear and linear forms, and applies boundary conditions.
Solving is done using the PETSc library.
"""

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
        """
        Initialize the MecSolver class.
        Creates the necessary forms for the incompatible plastic distortion and displacement fields. and calls the form compiler

        Args:
            mecFe (MecFE): The finite element object of the mechanics problem, which contains the finite element spaces and main functions
        """
        self.mecFe = mecFe
        self.create_forms_UPperp()
        self.create_forms_U()

    def create_forms_UPperp(self):
        """
        Create the bilinear and linear forms for the incompatible plastic distortion
        field :math:`\\mathbf{U_p}_{\perp}` for solving the div-curl system using the
        Least Squares Finite Element Method (LSFEM).

        The equations solved inside the body :math:`\Omega` are:

        .. math::

            \\nabla \cdot \\mathbf{U_p}^{\perp} = 0, \\quad 
            \\nabla \\times \\mathbf{U_p}^{\perp} = -\\boldsymbol{\\alpha}

        and, on the boundary :math:`\\partial \\Omega`:

        .. math::

            \\mathbf{U_p}^{\perp} \cdot n = 0
        These 3 equations are solved in the weak sense using the LSFEM, considering the following lagragian:

        .. math::
            \mathcal{L} = \\int_{\Omega} ||\\nabla \\cdot \\mathbf{U_p}^{\perp}||^2 \, d\Omega +\\int_{\Omega} ||\\nabla \\times \\mathbf{U_p}^{\perp} + \\boldsymbol{\\alpha}||^2 \, d\Omega + C_x \\int_{\\partial \\Omega} (\\mathbf{U_p}^{\perp} \cdot n)^2 \, dS

        Th corresponding bilinear form is defined as:

        
        .. math::
            a_{inc}(\\mathbf{u}_{inc}, \\mathbf{v}_{inc}) = \int_{\Omega} \\nabla \\times \\mathbf{u}_{inc} :\\nabla \\times \\mathbf{v}_{inc} + (\\nabla \cdot \\mathbf{u}_{inc}) \cdot (\\nabla \cdot \\mathbf{v}_{inc}) + C_x (\\mathbf{u}_{inc} \cdot n) (\\mathbf{v}_{inc} \cdot n) \, dS


        The linear form is defined as:

        
        .. math::

            L_{inc}(\\mathbf{v}_{inc}) =-\int_{\Omega} \\boldsymbol{\\alpha} : \\nabla \\times \\mathbf{v}_{inc} \, d\Omega

        where :math:`\\boldsymbol{\\alpha}` is the Dislocation density tensor,
        :math:`C_x` is a constant, and :math:`n` is the facet normal.

        The MPC cannot handle tensor fields, so we use formulate the problem in terms high dimensionner vector fields in the case of periodicity.
        """
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
        """
        Create the bilinear and linear forms for the elasticy problem using displacement field :math:`\\mathbf{U_e}`.  
        The equations solved inside the body :math:`\Omega` are:

        .. math::

            \\nabla \cdot \\mathbb{C}:\epsilon(u) = \\mathbf{f} + \\nabla \cdot \\mathbb{C} : \\mathbf{U_p} 

        the bilinear form is defined as:



        .. math::

            a_{u}(\\mathbf{u}, \\mathbf{v}) = \\int_{\Omega} \\mathbb{C} : \\epsilon(\\mathbf{u}) : \\epsilon(\\mathbf{v}) \, d\Omega 

        The linear form is defined as:


        .. math::


            L_{u}(\\mathbf{v}) = \\int_{\Omega} \\mathbb{C} : \\epsilon(\\mathbf{U_p}) : \\epsilon(\\mathbf{v}) \, d\Omega + C_w \\int_{\Omega} (\\mathbf{U_p}+\mathbf{Q})_{sym} : \\epsilon(\\mathbf{v}) \, d\Omega  + \\int_{\Omega} \\mathbf{f} : \\mathbf{v} \, d\Omega

        For now, Neumann boundary conditions are not implemented, so the linear form doesn't include any boundary term for.
        """
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
    

    def configure_solver_UPperp(self,bcs )->None:
        """
        Configure the solver for the incompatible plastic distortion field.
        It also assembles the bilinear form using the corresponding function spaces and boundary conditions. 

        Args:
            bcs (_type_): _description_
        """
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
            # Assemble the bilinear form for the incompatible plastic distortion field
            self.mecFe.A_inc = fem.petsc.assemble_matrix(self.mecFe.a_inc, bcs=bcs) #Allocate the matrix and initialize it
            self.mecFe.A_inc.assemble() # Assemble the matrix using petsc4py

            #Create a KSP solver for the incompatible plastic distortion field
            self.mecFe.problem_UPperp = PETSc.KSP().create(self.mecFe.domain.comm)
            self.mecFe.problem_UPperp.setOperators(self.mecFe.A_inc)
            self.mecFe.problem_UPperp.setInitialGuessNonzero(True)
            # Set the solver type and convergence options
            self.mecFe.problem_UPperp.setType(PETSc.KSP.Type.BCGS)
            self.mecFe.problem_UPperp.rtol = 1e-8
            self.mecFe.problem_UPperp.atol = 1e-10
            # self.mecFe.uPperpmoni = CVmonitor()
            # self.mecFe.problem_UPperp.setMonitor(self.mecFe.uPperpmoni.monitor)

            # Set the preconditioner type and options
            pc = self.mecFe.problem_UPperp.getPC()
            pc.setType(PETSc.PC.Type.GAMG)
            pc.setReusePreconditioner(True)



    def configure_solver_U(self,bcs)->None:
        """
        Configure the solver for elasticity problem.
        this method sets up the solver for the displacement vector field with the specified boundary conditions.

        It also assembles the bilinear form using the corresponding function spaces and boundary conditions. 

        Args:
            bcs (_type_): _description_
        """
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
                assert ns.test(self.mecFe.A_u)
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
        """
        Solve the div-curl system for the incompatible plastic distortion field.
        This method uses the previously configured solver to solve the system and applies the boundary conditions.
        It assembles the linear form into the right-hand side of the system using the boundary conditions.
        Then calls the solver to compute the plastic distortion field :math:`\\mathbf{U_p}^{\\perp}`.

        
        Args:
            bcs (_type_): _description_
        """
        if self.mecFe.periodic_UP:
            with dolfinx.common.Timer() as t_cpu:
                sol = self.mecFe.problem_UPperp.solve()
                print("Uperp done in : %s" % t_cpu.elapsed()[0])
                self.mecFe.U4.interpolate(fem.Expression(sol,self.mecFe.vector_sp4.element.interpolation_points()))
                # The solution is a vector field, we need to convert it to a tensor field
                self.mecFe.UPerpendicular.interpolate(fem.Expression(v2T(self.mecFe.U4,2),self.mecFe.tensor_sp2.element.interpolation_points()))
            # self.mecFe.uPperpmoni.plot()
            # self.mecFe.uPperpmoni.n+=1
        else:
            with dolfinx.common.Timer() as t_cpu:
                #Create a vector for the right-hand side of the system
                b_inc = fem.petsc.create_vector(self.mecFe.L_inc)
                #Set the vector to zero
                with b_inc.localForm() as loc_b:
                    loc_b.set(0)
                
                fem.petsc.assemble_vector(b_inc, self.mecFe.L_inc) # Assemble the linear form into the vector
                # Apply the boundary conditions to the vector
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
        """
        Solve the elasticity problem for the displacement vector field. 
        starts by assembling the linear form into the right-hand side of the system using the boundary conditions.
        Then calls the solver to compute the displacement field :math:`\\vec{u}`

        Then the displacement field is interpolated into :math:`\\vec{u}_{out}` for the output field.
        The distortion tensor field :math:`\mathbf{U} = \\nabla \\vec{u}` is also computed as the gradient of the displacement field.

        Args:
            bcs (_type_): _description_
        """
        if self.mecFe.periodic_u:
            with dolfinx.common.Timer() as t_cpu:
                soli = self.mecFe.problem_U.solve()
                print("U done in : %s" % t_cpu.elapsed()[0])
                self.mecFe.U.interpolate(fem.Expression(ufl.grad(soli),self.mecFe.tensor_sp2.element.interpolation_points()))
                self.mecFe.u_out.interpolate(fem.Expression(soli,self.mecFe.vector_sp2.element.interpolation_points()))
        else:
            with dolfinx.common.Timer() as t_cpu:
                # Create a vector for the right-hand side of the system
                with self.mecFe.b_u.localForm() as loc_b:
                    loc_b.set(0)
                # Assemble the linear form into the vector
                fem.petsc.assemble_vector(self.mecFe.b_u, self.mecFe.L_u)

                # Apply the boundary conditions to the vector
                if len(bcs)>0:
                    fem.petsc.apply_lifting(self.mecFe.b_u, [self.mecFe.a_u], bcs=[bcs])
                    self.mecFe.b_u.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    for bc in bcs:
                        bc.set(self.mecFe.b_u.array_w)
                else:
                   self.mecFe.b_u.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                # if the nullspace is added, we need to remove the nullspace from the right-hand side vector
                if self.mecFe.mech_params.addNullspace:
                    self.mecFe.A_u.getNullSpace().remove(self.mecFe.b_u)
                # Solve the system
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
