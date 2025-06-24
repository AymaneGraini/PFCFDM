"""
Mechanical Computation class for computing auxiliary fields in a mechanical simulation.
This class computes fields that are not directly part of the variational formulation 
such as stresses, curls, divergences, velocities, and strains.
This module is part of a larger simulation framework and relies on the MecFE class
"""

from Simulation.Parameters import *
import dolfinx.fem as fem
import ufl
from utils.MyAlgebra import *
from .MecFE import *

class MecComp:
    """
        A class to compute auxiliary fields doesn't appear directly in 
        the variational formulation but are used for post-processing

        Args:
            mecFE (MecFE): An instance of the MecFE class that contains the finite element spaces and other parameters.
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

        self.curlUE   = fem.Function(mecFE.tensor_sp3,name="curlUe")
        self.curlUP   = fem.Function(mecFE.tensor_sp3,name="curlUp")
        self.curlQ   = fem.Function(mecFE.tensor_sp3,name="curlQ")

    def compute_sym(self):
        """
            Compute the symmetric part of the fields
            :math:`sym(\\mathbf{U_e})` and :math:`sym(\\mathbf{Q})`
        """
        self.mecFE.UEsym.interpolate(fem.Expression(ufl.sym(self.mecFE.UE),self.mecFE.tensor_sp2.element.interpolation_points()))
        self.Qsym.interpolate(fem.Expression(ufl.sym(self.mecFE.Q),self.mecFE.tensor_sp2.element.interpolation_points()))

    def compute_curls(self):
        """
            Compute the curl of the fields :math:`\\nabla \\times \\mathbf{U_e}`,
            :math:`\\nabla \\times \\mathbf{U_p}` and :math:`\\nabla \\times \\mathbf{Q}`
        """
        self.curlUE.interpolate(fem.Expression(tcurl(extendT(self.mecFE.UE)),self.mecFE.tensor_sp3.element.interpolation_points()))
        self.curlUP.interpolate(fem.Expression(tcurl(extendT(self.mecFE.UP)),self.mecFE.tensor_sp3.element.interpolation_points()))
        self.curlQ.interpolate(fem.Expression(tcurl(extendT(self.mecFE.Q)),self.mecFE.tensor_sp3.element.interpolation_points()))

    def compute_stresses(self):
        """
            Compute the stress due to :math:`\\mathbf{Q}` as :math:`\\mathbb{C}:sym\\mathbf{Q}`
            and the elastic stress either:
                un-Coupled: :math:`\\mathbb{C}:sym(\\mathbf{U_e})`
                or Coupled : :math:`\\mathbb{C}:sym(\\mathbf{U_e}) + C_w \\, sym(\\mathbf{U_e}-\\mathbf{Q})`
            Results are stored in self.sigmaUe and self.sigmaQ
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
            Compute the divergence of both stresses  :math:`\\mathbf{div}(\sigma_{U_e})` and :math:`\\mathbf{div}(\sigma_Q)`
            coming from self.compute_stresses() 

            Results are stored in self.divsUe and self.divsQ
           
        """

        self.divsUe.interpolate(fem.Expression(ufl.div(self.sigmaUe), self.mecFE.vector_sp2.element.interpolation_points()))
        self.divsQ.interpolate(fem.Expression(ufl.div(self.sigmaQ), self.mecFE.vector_sp2.element.interpolation_points()))

    def compute_velocity(self):
        """
            Compute the classical Peach-Khoeler force without considering the coupling term.
            The Peach-Khoeler force is given by the equation:
            :math:`\\mathbf{V}_{pk} = (\sigma_{e} \cdot \\alpha)^t:\\mathbf{X}`
            where :math:`\sigma_{e}` is the stress due to the elastic deformation, and :math:`{X}` is the 3rd order Levi-Civita tensor.
            Given the current alpha and sigma

            Generates a vector field stored in self.V_pk
        """
        Cw      = self.sim_params.Cw
        e=5
        i, j,k, l = ufl.indices(4)
        # V_pk_ufl = V_pk_ufl = ufl.as_vector(tuple(
        #         (extendT(self.sigmaUe)[i,k] +Cw*ufl.transpose(extendT(self.mecFE.UE))[i,k]-Cw*ufl.transpose(extendT(self.mecFE.Q))[i,k]) * self.mecFE.alpha[k,j] * perm[i,j,l]
        #         for l in range(3)  
        #     ))
        V_pk_ufl = V_pk_ufl = ufl.as_vector(tuple(
                (extendT(self.sigmaUe)[i,k] + e*ufl.transpose(tcurl(self.mecFE.alpha))[i,k]) * self.mecFE.alpha[k,j] * perm[i,j,l]
                for l in range(3)  
            ))
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