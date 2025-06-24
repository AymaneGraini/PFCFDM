from dataclasses import dataclass, field
import dolfinx.fem as fem
import numpy as np

@dataclass
class MechParams:
    """A class for organizing all the parameters of the mechanics part
    The parameters are:
    -   lambda_     : Lame's first parameter
    -   mu          : Lame's second parameter (shear modulus)
    -   Cx          : coefficient of boundary condition penalty in the LSFEM formulation of the div-curl system for solving :math:`\mathbf{U_p}^\perp`
    -   Cel         : coefficient of the elastic energy in the mechanical problem
    -   f           : applied body force vector
    -   periodic_UP : whether the simulation is periodic in the UP space or not
    -   periodic_u  : whether the simulation is periodic in the u space or not
    -   addNullspace: whether to add the nullspace to the elasticity problem space or not (useful in pure-neumann problems)
    """
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
    """A class for organizing all the parameters of the phase field part
    The parameters are    : 
    -   a0                : lattice constant
    -   qs                : list of 1st mode wavevectors of the desired crystal
    -   ps                : list of 2nd mode wavevectors of the desired crystal
    -   r                 : PFC main parameter (must be >0)
    -   avg               : average value of Psi
    -   periodic          : whether the simulation is periodic or not
    -   deg               : degree of the evolution equation 4 or 6
    -   motion            : the way to compute the dislocation current either through J (up) or via the dislocation velocity (v) (This doesn't work yet)
    -   write_amps        : whether to write the amplitudes of the modes to file or not
    -   ConservationMethod: method to conserve the total average, either "LM" or "scalar" or "field"
    -   q0                : magnitude of the 1st mode wavevectors (not used for now)
    """
    a0                : float
    qs                : np.ndarray
    ps                : np.ndarray
    r                 : float
    avg               : float
    periodic          : bool
    deg               : int
    motion            : str
    write_amps        : bool
    ConservationMethod: str
    q0                : float = 1  # This q0 is not used for now TODO

@dataclass
class SimParams:
    """ 
        A class that contains all information about 
        the simulation nature and parameters
        The parameters are:
        -   Csh        : coefficient of the swift-hohenberg in the evolution equation
        -   Cw         : coefficient of the penality term in the evolution equation
        -   penalty_Psi: whether to add the penalty term in the evolution equation of Psi or not
        -   penalty_u  : whether to add the penalty term in the definition of the elastic stress
        -   dt         : time step
        -   tmax       : maximum time for the simulation
        -   outFreq    : frequency of the output
        -   L          : length of the domain
        -   H          : height of the domain
    """
    Csh        : float
    Cw         : float
    penalty_Psi: bool
    penalty_u  : bool
    dt         : float
    tmax       : float
    outFreq    : int
    L          : float
    H          : float

@dataclass
class GeomParams:
    """A class that contains all geometry
    The parameters are:
    -   dx : grid spacing in the x direction
    -   dy : grid spacing in the y direction
    -   Nx : number of grid points in the x direction
    -   Ny : number of grid points in the y direction
    The length L and height H of the domain are automatically computed in the __post_init__ method.
    """
    dx: float
    dy: float
    Nx: float
    Ny: float
    L : float = field(init=False)
    H : float = field(init=False)

    def __post_init__(self):
        self.L = self.dx * self.Nx
        self.H = self.dy * self.Ny