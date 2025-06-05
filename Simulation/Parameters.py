from dataclasses import dataclass, field
import dolfinx.fem as fem
import numpy as np

@dataclass
class MechParams:
    """A class for organizing all the parameters of the mechanics part"""
    lambda_     : float
    mu          : float
    Cx          : float
    Cel         : float
    f           : np.ndarray
    periodic_UP : bool
    periodic_u  : bool
    addNullspace: bool

@dataclass
class PfcParams:
    """A class for organizing all the parameters of the phase field part"""
    a0        : float
    qs        : np.ndarray
    ps        : np.ndarray
    r         : float
    avg       : float
    periodic  : bool
    deg       : int
    motion    : str
    write_amps: bool
    q0        : float = 1  # This q0 is not used for now TODO

@dataclass
class SimParams:
    """ 
        A class that contains all information about 
        the simulation nature and parameters
    """
    Csh        : float
    Cw         : float
    penalty_Psi: bool
    penalty_u  : bool
    dt         : float
    tmax       : float
    L          : float
    H          : float

@dataclass
class GeomParams:
    """A class that contains all geometry"""
    dx: float
    dy: float
    Nx: float
    Ny: float
    L : float = field(init=False)
    H : float = field(init=False)

    def __post_init__(self):
        self.L = self.dx * self.Nx
        self.H = self.dy * self.Ny