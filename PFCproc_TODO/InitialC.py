import numpy as np

def initialize_from_burgers(qs,ps,list_defects,A,avg):
    def mdot(q,x):
        return q[0]*x[0]+q[1]*x[1]

    def disp(q,defect,x):
        xp,yp,bs=defect
        return (1/(2*np.pi))*mdot(q,bs)*np.arctan2((x[1]-yp),(x[0]-xp))

    def disp_Anderson_edge(q,defect,x):
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
            result += np.exp(1j*mdot(q,x) + 1j*inner_sum)
        for p in ps:
            inner_sum = sum(disp(p,defect,x) for defect in list_defects) # Could be disp or disp anderson
            result += 0.5*np.exp(1j*mdot(p,x) + 1j*inner_sum)
        return np.real(result)
    
    return lambda x: avg+A*f(x)