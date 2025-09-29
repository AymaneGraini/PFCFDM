import numpy as np
from typing import Callable

def mdot(q:np.ndarray,x:np.ndarray)->float:
        """
        Dot product between 2 ndarrays
        A
        """
        return q[0]*x[0]+q[1]*x[1]

def initialize_from_burgers(qs:np.ndarray,ps:np.ndarray,list_defects:np.ndarray,A:float,avg:float) -> Callable:
    """
    Creates a lambda function that computes :math:psi(x) for a given x
    g : float is the shear
    """

    def disp(q,defect,x):
        xp,yp,bs=defect
        H=np.max(x[1])
        return (1/(2*np.pi))*mdot(q,bs)*np.arctan2((x[1]-yp),(x[0]-xp))

    def disp_Anderson_edge(q,defect,x):
        """Displacement field coming from elastic theory of dislocation

        """
        xp,yp,bs=defect
        v=1/4.
        u=np.array([
            np.arctan2((x[1]-yp),(x[0]-xp))+(1/(2*(1-v)))*(x[1]-yp)*(x[0]-xp)/((x[1]-yp)**2+(x[0]-xp)**2),
            -((1-2*v)/(4-4*v))*np.log((x[1]-yp)**2+(x[0]-xp)**2) + (1/(4-4*v))*((x[1]-yp)**2-(x[0]-xp)**2)/((x[1]-yp)**2+(x[0]-xp)**2)
            ])
        return (1/(2*np.pi))*bs[0]*mdot(q,u) 

    def f(x):
        result = 0
        for q in qs:
            inner_sum = sum(disp(q,defect,x) for defect in list_defects) # Could be disp or disp anderson
            result += np.exp(1j*mdot(q,x) + 1j*inner_sum) #SHould this be + sum or - sum ? #TODO
        for p in ps:
            inner_sum = sum(disp(p,defect,x) for defect in list_defects) # Could be disp or disp anderson
            result += 0.5*np.exp(1j*mdot(p,x) + 1j*inner_sum)
        return np.real(result)
    
    return lambda x: avg+A*f(x)


def dislocation_loop(qs:np.ndarray,ps:np.ndarray,b:np.ndarray,r0:np.ndarray,R:float,A:float,avg:float) -> Callable:
    x0,y0= r0
    w=1
    def disp(q,x):
        r = np.hypot(x[0]-x0, x[1]-y0)
        sn = (1/(2*np.pi))*mdot(q,b)
        H = 0.5*(1 - np.tanh((r-R)/w))
        return sn*H
    
    def f(x):
        result = 0
        for q in qs:
            inner_sum = disp(q,x) # Could be disp or disp anderson
            result += np.exp(1j*mdot(q,x) + 1j*inner_sum) #SHould this be + sum or - sum ? #TODO
        return np.real(result)
    
    return lambda x: avg+A*f(x)
