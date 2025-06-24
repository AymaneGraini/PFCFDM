
from dataclasses import dataclass
import numpy as np


@dataclass
class Lattice:
    """A class to represent a lattice in 2D space.
    parameters are:
    - qs : np.ndarray
        An array of wavevectors corresponding to the first mode of the lattice.
    - ps : np.ndarray
        An array of wavevectors corresponding to the second mode of the lattice.
    """
    qs : np.ndarray
    ps : np.ndarray


hex_lat = Lattice(np.array([[0,1],
                            [np.sqrt(3)/2,-1/2],
                            [-np.sqrt(3)/2,-1/2],
                            [0,-1],
                            [-np.sqrt(3)/2,1/2],
                            [np.sqrt(3)/2,1/2]]),np.array([]))

square_lat = Lattice(np.array([[0,1],
                               [1,0],
                               [0,-1],
                               [-1,0]]),
                    np.array([[1,-1],
                                [1,1],
                                [-1,1],
                                [-1,-1]]))