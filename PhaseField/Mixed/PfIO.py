import dolfinx.io
from .PfFe import *
from .PfComp import *
import numpy as np

class PfIO:
    """
        A Mechanics related class, used to write output in XDMF
        Also writes different relevant indicators, see below.
    """
    def __init__(self,
                 pfFe: PFFe,
                 pfComp: PfComp,
                 file : dolfinx.io.XDMFFile
                 ):
        self.file = file
        self.pfFe=pfFe
        self.pfComp=pfComp
    


    def write_output(self,t):
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
