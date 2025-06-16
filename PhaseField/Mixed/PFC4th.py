import numpy as np
import dolfinx
import basix
import dolfinx.fem as fem
import ufl
import scipy.ndimage as ndimage
from .PfFe import *



class PFC4(PFFe):
    def __init__(self,
                 domain:dolfinx.mesh.Mesh,
                 pfc_params: PfcParams,
                 sim_params: SimParams
                 )->None:
        super().__init__(domain,pfc_params,sim_params)
        self.set_spaces()
        self.set_funcs()
        self.set_projection_basis()
        self.pbcs = PeriodicBC_geometrical(domain, self.MEl_space,2,[]) if self.pfc_params.periodic else None


    def set_spaces(self):
        self.elem = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1)
        self.MEl_space = fem.functionspace(self.domain, basix.ufl.mixed_element([self.elem, self.elem]))
        super().set_spaces()

    def set_funcs(self):
        super().set_funcs()
        self.psi0, self.chi0 = ufl.split(self.zeta0)
        (self.psi_current, self.chi_current) = ufl.TrialFunctions(self.MEl_space)
        self.q, self.v = ufl.TestFunctions(self.MEl_space)
        self.corr=  fem.Function(self.MEl_space,name="Correction")

    def create_forms(self):
        r=self.pfc_params.r
        dt=self.sim_params.dt
        Csh=self.sim_params.Csh
        Cw =self.sim_params.Cw

        self.psi0, self.chi0 = ufl.split(self.zeta0)
        self.a_pfc =fem.form(
            (1/(dt*Csh)+1-r)*ufl.inner(self.psi_current,self.q)*self.dx
            -ufl.inner(ufl.grad(self.chi_current),ufl.grad(self.q))*self.dx
            +2*ufl.inner(self.chi_current,self.q)*self.dx
            +ufl.inner(self.chi_current,self.v)*self.dx+ufl.inner(ufl.grad(self.psi_current),ufl.grad(self.v))*self.dx)

        if Cw==0:
            print("Iiniting with Cw=0")
            self.L_pfc =fem.form(
                ufl.inner((1/(dt*Csh))*self.psi0-self.psi0**3,self.q)*self.dx
            )
        else:
            print("Iiniting with Cw=",Cw)
            self.L_pfc =fem.form(
                ufl.inner((1/(dt*Csh))*self.psi0-self.psi0**3-(Cw/Csh)*self.dFQW,self.q)*self.dx
            )
        self.L_chi      = fem.form(-1.0*ufl.inner(ufl.grad(self.psi0),ufl.grad(self.v))*self.dx)
        self.Energyform = fem.form(((1/2)*(1-r)*self.psi0**2+(1/4)*self.psi0**4+
                                    (1/2)*self.chi0**2-ufl.inner(ufl.grad(self.psi0),ufl.grad(self.psi0)))
                                    *self.dx)
        self.Avg_form = fem.form(self.SH_sol.sub(0)*self.dx)

    def correct(self):
        # psi_avg = fem.assemble_scalar(self.Avg_form)/(self.sim_params.L*self.sim_params.H)
        # self.corr.sub(0).interpolate(lambda x: x[0]*0.0+(self.pfc_params.avg-psi_avg))
        # self.SH_sol.x.array[:] += self.corr.x.array

        beta = (self.psiout.x.petsc_vec.dot(self.b_basis.x.petsc_vec)-self.pfc_params.avg)/self.b_basis_norm
        self.psiout.x.petsc_vec.axpy(-1.0*beta, self.b_basis.x.petsc_vec)
        self.psiout.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
        self.SH_sol.sub(0).interpolate(self.psiout)


        self.zeta0.interpolate(self.SH_sol)
        self.SH_sol.x.scatter_forward()
        self.zeta0.x.scatter_forward()
        self.psi0, self.chi0 = ufl.split(self.zeta0)
        self.psiout.interpolate(self.zeta0.sub(0))

