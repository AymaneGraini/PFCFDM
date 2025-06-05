import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)



# a=50

# cw=2
# root = "boostmech"+str(cw)

# filename=root+".h5"

# qi=2

def get_im_data(filename,fieldname,ti,comp=None):
    data1=None
    X,Y= None,None
    with h5py.File(filename, "r") as f:
        try :
            full_field_timeseries = f["Function/"+fieldname+"/"]
        except:
            raise ValueError("Requested field is not available. Available are", list(f["Function"]))

        timekeys= full_field_timeseries.keys()
        sorted_timesteps = sorted(timekeys, key=lambda s: float(s.replace("_", ".")))
        # print("exporting data from ", sorted_timesteps[ti])
        # print("ti = ", ti)
        if ti >= len(sorted_timesteps):
            raise ValueError("Requested timestep is not available, maximum snapshot is", len(sorted_timesteps)-1)
        Full_field = f["Function/"+fieldname+"/"+sorted_timesteps[ti]]
        mesh= f["Mesh/mesh/geometry"]
        x = mesh[:, 0]
        y = mesh[:, 1]
        nx = len(np.unique(x))
        ny = len(np.unique(y))
        newx = np.linspace(x.min(),x.max(),nx)
        newy = np.linspace(y.min(),y.max(),ny)
        X,Y = np.meshgrid(newx,newy) 
        mapping = np.lexsort((mesh[:, 0], mesh[:, 1]))

        if comp==None:
            field=Full_field[:]
        else:
            field=Full_field[:,comp]
            
        data1= field[mapping].reshape(ny, nx)
    timest = float(sorted_timesteps[ti].replace("_","."))
    return X,Y,data1, timest




# t=485

# filename="TEST_1_0.0_1_0.01.h5"
# X,Y,data1,tt= get_im_data(filename,"Psi",0)
# X_flat = X.flatten()
# Y_flat = Y.flatten()
# F_flat = data1.flatten()

# data = np.column_stack((X_flat, Y_flat, F_flat))

# np.savetxt('whydfdq_notsym'+str(t)+'.csv', data, delimiter='\t')