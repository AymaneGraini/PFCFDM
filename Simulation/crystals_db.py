
from dataclasses import dataclass
import numpy as np

#TODO docume,t

@dataclass
class Lattice:
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
                               [-1,0]]),np.array([[1,-1],
                                                  [1,1],
                                                  [-1,1],
                                                  [-1,-1]]))