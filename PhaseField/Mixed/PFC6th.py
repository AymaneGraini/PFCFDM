import numpy as np
import dolfinx
import basix
import dolfinx.fem as fem
import ufl
import scipy.ndimage as ndimage
from .PfFe import *

class PFC6(PFFe):
    def __init__(self,
                 domain:dolfinx.mesh.Mesh,
                 pfc_params,
                 sim_params
                 )->None:
        super().__init__(domain,pfc_params,sim_params)
        self.set_spaces()
        self.set_funcs()
        self.set_projection_basis()
        self.pbcs = PeriodicBC_geometrical(domain, self.MEl_space,3,[]) if self.pfc_params.periodic else None

        
    def set_spaces(self):
        self.elem = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1) 
        self.MEl_space = fem.functionspace(self.domain, basix.ufl.mixed_element([self.elem, self.elem,self.elem]))
        super().set_spaces()

    def set_funcs(self):
        super().set_funcs()
        self.psi0, self.mu0,self.chi0 = ufl.split(self.zeta0)
        (self.psi_current, self.mu_current,self.chi_current) = ufl.TrialFunctions(self.MEl_space)
        self.u, self.v,self.w = ufl.TestFunctions(self.MEl_space)

    def create_forms(self):
        r=self.pfc_params.r
        dt=self.sim_params.dt
        Csh=self.sim_params.Csh
        Cw =self.sim_params.Cw

        self.psi0, self.mu0,self.chi0 = ufl.split(self.zeta0)
        self.a_pfc =fem.form(ufl.inner(self.psi_current,self.u)*self.dx
                             +dt*Csh*ufl.inner(ufl.grad(self.mu_current),ufl.grad(self.u))*self.dx
                                +ufl.inner(self.mu_current,self.v) *self.dx+ ufl.inner(ufl.grad(self.chi_current),ufl.grad(self.v))*self.dx
                                +2*ufl.inner(ufl.grad(self.psi_current),ufl.grad(self.v))*self.dx
                                -(1-r)*ufl.inner(self.psi_current,self.v)*self.dx
                            
                             +ufl.inner(self.chi_current,self.w)*self.dx+ufl.inner(ufl.grad(self.psi_current),ufl.grad(self.w))*self.dx)
        
        if Cw==0:
            print("Initing with no coupling PFC-FDM in PFC evolution")
            self.L_pfc =fem.form(ufl.inner(self.psi0,self.u)*self.dx+(ufl.inner(self.psi0**3,self.v)*self.dx))
        else:
            print("Initing with Cw = ",Cw)

            self.L_pfc =fem.form(ufl.inner(self.psi0,self.u)*self.dx+(ufl.inner(self.psi0**3+Cw*self.dFQW,self.v)*self.dx))

        self.Energyform = fem.form(((1/2)*(1-r)*self.psi0**2+(1/4)*self.psi0**4+
                                    (1/2)*self.chi0**2-ufl.inner(ufl.grad(self.psi0),ufl.grad(self.psi0)))
                                    *self.dx)
        self.Avg_form = fem.form((1/(self.sim_params.L*self.sim_params.H))*self.SH_sol.sub(0)*self.dx)

    def correct(self):
        self.zeta0.x.array[:] =self.SH_sol.x.array
        self.SH_sol.x.scatter_forward()
        self.zeta0.x.scatter_forward()
        self.psi0, self.mu0,self.chi0 = ufl.split(self.zeta0)
        self.psiout.interpolate(self.zeta0.sub(0))


