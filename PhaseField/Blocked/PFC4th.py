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
        self.pbcs = [PeriodicBC_geometrical_nest(domain, self.main_space,1,[])  for _ in range(2)] if self.pfc_params.periodic else None #TODO Does it work in a blocked monolithic problem 

        self.maps = [(self.main_space.dofmap.index_map, self.main_space.dofmap.index_map_bs),
                     (self.main_space.dofmap.index_map, self.main_space.dofmap.index_map_bs),
                      (self.real_space.dofmap.index_map, self.real_space.dofmap.index_map_bs)]
        
        self.sizes = [imap.size_local * bs for imap, bs in self.maps]
        self.offsets=[0]
        for size in self.sizes[:-1]:
            self.offsets.append(self.offsets[-1] + size)
    def set_spaces(self):
        self.elem = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1)
        self.main_space = fem.functionspace(self.domain, self.elem)
        super().set_spaces()

    def set_funcs(self):
        super().set_funcs()
        self.psi0 = fem.Function(self.main_space) 
        self.lmbda = fem.Function(self.real_space)  #Lagrange multiplier
        self.chi0 = fem.Function(self.main_space,name="chi") 
        self.psi_sol = fem.Function(self.main_space) #Solved for (current sol)
        self.chi_sol = fem.Function(self.main_space) #Solved for (current sol)
        (self.psi_current, self.chi_current) =  ufl.TrialFunction(self.main_space), ufl.TrialFunction(self.main_space)
        self.q, self.v = ufl.TestFunction(self.main_space), ufl.TestFunction(self.main_space)
        self.dl = ufl.TestFunction(self.real_space)
        self._lm = ufl.TrialFunction(self.real_space)

    def create_forms(self):
        r          = self.pfc_params.r
        dt         = self.sim_params.dt
        Csh        = self.sim_params.Csh
        Cw         = self.sim_params.Cw


        self.a11    = (1/(dt*Csh)+1-r)*ufl.inner(self.psi_current,self.q)*self.dx
        self.a12    = -1.0*ufl.inner(ufl.grad(self.chi_current),ufl.grad(self.q))*self.dx+2*ufl.inner(self.chi_current,self.q)*self.dx
        self.a13    = -1.0*ufl.inner(self._lm,self.q)*self.dx

        self.a21    = ufl.inner(ufl.grad(self.psi_current),ufl.grad(self.v))*self.dx
        self.a22    = ufl.inner(self.chi_current,self.v)*self.dx
        self.a23    = None

        self.a31    = ufl.inner(self.psi_current,self.dl)*self.dx
        self.a32    = None
        self.a33    = None

        self.a_pfc  = fem.form([[self.a11,self.a12,self.a13],
                       [self.a21,self.a22,self.a23],
                       [self.a31,self.a32,self.a33]])

        if Cw==0:
            print("Iiniting with Cw=0")
            self.L_pfc = fem.form([ ufl.inner((1/(dt*Csh))*self.psi0-self.psi0**3,self.q)*self.dx,
                                   ufl.inner(fem.Constant(self.domain, PETSc.ScalarType(0)), self.v) * self.dx,
                                   ufl.inner(fem.Constant(self.domain, PETSc.ScalarType(0)), self.dl) * self.dx
                                   ]) 
                                   
        else:
            self.L_pfc = fem.form([ ufl.inner((1/(dt*Csh))*self.psi0-self.psi0**3-(Cw/Csh)*self.dFQW,self.q)*self.dx,
                                    ufl.inner(fem.Constant(self.domain, PETSc.ScalarType(0)), self.v) * self.dx,
                                   ufl.inner(fem.Constant(self.domain, PETSc.ScalarType(0)), self.dl) * self.dx
                                    ]) 


        self.L_chi      = fem.form(-1.0*ufl.inner(ufl.grad(self.psi0),ufl.grad(self.v))*self.dx)
        self.Energyform = fem.form(((1/2)*(1-r)*self.psi0**2+(1/4)*self.psi0**4+
                                    (1/2)*self.chi0**2-ufl.inner(ufl.grad(self.psi0),ufl.grad(self.psi0)))
                                    *self.dx)
        self.Avg_form = fem.form((1/(self.sim_params.L*self.sim_params.H))*self.psi0*self.dx)

    def correct(self):

        # beta = (self.psiout.x.petsc_vec.dot(self.b_basis.x.petsc_vec)-self.pfc_params.avg)/self.b_basis_norm
        # self.psiout.x.petsc_vec.axpy(-1.0*beta, self.b_basis.x.petsc_vec)

        # psi_avg = fem.assemble_scalar(fem.form(self.psiout*ufl.dx))/(self.sim_params.L*self.sim_params.H)
        # self.psiout.x.array[:] = self.psiout.x.array +self.pfc_params.avg - psi_avg


        # self.psi0.lam.x.array[:sizes[2]] = x_array[offsets[2]:]
        # self.psiout.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
        self.psiout.interpolate(self.psi_sol)
        self.psi0.interpolate(self.psi_sol)
        self.chi0.interpolate(self.chi_sol)
        self.psi0.x.scatter_forward()
        self.chi0.x.scatter_forward()

