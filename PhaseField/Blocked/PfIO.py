"""
A class, used to write output in XDMF
Also writes different relevant indicators, see below.
"""
import dolfinx.io
from .PfFe import *
from .PfComp import *
import numpy as np

class PfIO:
    """
        A class, used to write output in XDMF
        Also writes different relevant indicators, see below.
    """
    def __init__(self,
                 pfFe: PFFe,
                 pfComp: PfComp,
                 file : dolfinx.io.XDMFFile
                 ):
        
        """
        Initializes the PfIO class.

        Args:
            pfFe (PFFe): An instance of the PFFe class containing phase field equations.
            pfComp (PfComp): An instance of the PfComp class containing auxiliary parameters.
            file (dolfinx.io.XDMFFile): The XDMF file to which output will be written.  
        """
        self.file = file
        self.pfFe=pfFe
        self.pfComp=pfComp
    


    def write_output(self,t):
        """
        Writes the output to the XDMF file at time t.

        Args:
            t (float): The current time step for which the output is written.
        """
        self.file.write_function(self.pfFe.psiout,t)
        self.file.write_function(self.pfFe.dFQW,t)
        self.file.write_function(self.pfComp.alphaT,t)
        self.file.write_function(self.pfComp.J,t)
        # self.file.write_function(self.velocity,t)
        # self.file.write_function(self.micro_sigma,t)
        # self.file.write_function(self.micro_sigma_avg,t)
        if self.pfFe.pfc_params.write_amps:
            for i,q in enumerate(self.pfFe.pfc_params.qs):
                self.file.write_function(self.pfFe.Re_amps[i],t)
                self.file.write_function(self.pfFe.Im_amps[i],t)
   
    def writepos(self):
        pos_array = np.array(self.pos, dtype=object)
        np.savetxt("positions.csv", pos_array, delimiter=",", fmt="%s")
