from Simulation.Parameters import *
import dolfinx.fem as fem
import ufl
from utils.MyAlgebra import *
from .MecFE import *

class MecComp:
    """
        A class to compute auxiliary fields doesn't appear directly in 
        the variational formulation but are used for post-processing
    """
    def __init__(self,
                 mecFE : MecFE):
        
        self.mecFE = mecFE
        self.mech_params = mecFE.mech_params
        self.sim_params = mecFE.sim_params
        self.set_funcs(mecFE)

    def set_funcs(self,
                  mecFE : MecFE):
        """
            Define all the FE functions for each relevant field 
            using the FE spaces stored in mecFE
        """
        self.sigmaUe = fem.Function(mecFE.tensor_sp2,name="sigmaUe")
        self.sigmaQ  = fem.Function(mecFE.tensor_sp2,name="sigmaQ")

        self.divsUe = fem.Function(mecFE.vector_sp2,name="div_sUe")
        self.divsQ  = fem.Function(mecFE.vector_sp2,name="div_sQ")

        self.V_pk   = fem.Function(mecFE.vector_sp3,name="VelocityPK")
        self.epsilon_psi = fem.Function(mecFE.tensor_sp2,name="epsilonPsi")

        self.Qsym        = fem.Function(mecFE.tensor_sp2,name="Qsym")

    def compute_sym(self):
        self.mecFE.UEsym.interpolate(fem.Expression(ufl.sym(self.mecFE.UE),self.mecFE.tensor_sp2.element.interpolation_points()))
        self.Qsym.interpolate(fem.Expression(ufl.sym(self.mecFE.Q),self.mecFE.tensor_sp2.element.interpolation_points()))

    def compute_stresses(self):
        """
            Compute the stress due to Q as C:sym(Q)
            and the elastic stress either:
                non Coupled: C:sym(Ue) 
                or Coupled : C:sym(Ue) + Cw*sym(Ue-Q)
        """
        lambda_ = self.mech_params.lambda_
        mu_     = self.mech_params.mu
        Cw      = self.sim_params.Cw
        Cel     = self.mech_params.Cel

        if self.sim_params.penalty_u:
            self.sigmaUe.interpolate(fem.Expression(Cel*sigma(ufl.sym(self.mecFE.UE),lambda_,mu_)
                                                    +Cw*ufl.sym(self.mecFE.UE-self.mecFE.Q), 
                                                    self.mecFE.tensor_sp2.element.interpolation_points()))
        else:
            self.sigmaUe.interpolate(fem.Expression(Cel*sigma(ufl.sym(self.mecFE.UE),lambda_,mu_), 
                                                    self.mecFE.tensor_sp2.element.interpolation_points()))

        self.sigmaQ.interpolate(fem.Expression(Cel*sigma(ufl.sym(self.mecFE.Q),lambda_,mu_), 
                                               self.mecFE.tensor_sp2.element.interpolation_points()))

    
    def compute_divergence(self):
        """
            Compute the divergence of both stresses
            coming from self.compute_stresses()
        """
        self.divsUe.interpolate(fem.Expression(ufl.div(self.sigmaUe), self.mecFE.vector_sp2.element.interpolation_points()))
        self.divsQ.interpolate(fem.Expression(ufl.div(self.sigmaQ), self.mecFE.vector_sp2.element.interpolation_points()))

    def compute_velocity(self):
        """
            Compute the classical Peach-Khoeler force
            Given the current alpha and sigma
            generates a vector field.
        """
        i, j,k, l = ufl.indices(4)
        V_pk_ufl = ufl.as_vector(extendT(self.sigmaUe)[i,k]*self.mecFE.alpha[k,j]*perm[i,j,l],(l,))
        self.V_pk.interpolate(fem.Expression(V_pk_ufl, self.mecFE.vector_sp3.element.interpolation_points()))

    def Compute_Epsilon_Psi(self,pfcSolver):
        """
            Computes the strain coming from
            the coarse-grained microscopic stress
            from the phase field psi.
        """
        lambda_ = self.mech_params.lambda_
        mu_     = self.mech_params.mu
        self.epsilon_psi.interpolate(fem.Expression(strain(pfcSolver.micro_sigma_avg,lambda_,mu_)
                                                    ,self.mecFE.tensor_sp2.element.interpolation_points()))