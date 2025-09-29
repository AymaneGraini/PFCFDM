



import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.ndimage as ndimage
def delta(Z,w):
    return (1/(2*np.pi*w**2))*np.exp(-(Z**2)/(2*w**2))


a0 = 4*np.pi/np.sqrt(3)
xmin, xmax = 150, 280
ymin, ymax = 50,120


filename="out/coupled/debug/AmitJDipole_0.1_0_-0.1_0.h5"

def get_defect_coords(filename):
    with h5py.File(filename, "r") as f:
            defect1_loc=[]
            defect2_loc=[]
            ts=[]
            full_field_timeseries = f["Function/Psi/"]
            timekeys= full_field_timeseries.keys()
            sorted_timesteps = sorted(timekeys, key=lambda s: float(s.replace("_", ".")))

            mesh= f["Mesh/mesh/geometry"]
            x = mesh[:, 0]
            y = mesh[:, 1]
            nx = len(np.unique(x))
            ny = len(np.unique(y))
            newx = np.linspace(x.min(),x.max(),nx)
            newy = np.linspace(y.min(),y.max(),ny)
            X,Y = np.meshgrid(newx,newy) 
            mapping = np.lexsort((mesh[:, 0], mesh[:, 1]))
            dx = np.abs(X[0, 1] - X[0, 0])
            dy = np.abs(Y[1, 0] - Y[0, 0])
            dA = dx * dy
            # mapping = np.lexsort((mesh[:, 0], mesh[:, 1]))

            mapping = np.lexsort((mesh[:, 0], mesh[:, 1]))
            field = f["Function/dFQW/"+sorted_timesteps[-1]][:,0]
            df = field[mapping].reshape(ny, nx)   
            fieldU = f["Function/UE/"+sorted_timesteps[-1]][:,1]
            fieldQ = f["Function/Q/"+sorted_timesteps[-1]][:,1]
            U = fieldU[mapping].reshape(ny, nx)   
            Q = fieldQ[mapping].reshape(ny, nx)   
    return X,Y,df,U,Q




files=[
# "out/debugpenalty/Pen0.1_0_0.h5",
# "out/debugpenalty/Pen0.1_0.1_0.05.h5",
# "out/debugpenalty/Pen0.1_1_0.05.h5",
"out/debugpenalty/Pen0.1_0.01_0.05.h5",
# "out/debugpenalty/Pen0.1_10_0.05.h5",
]
colors=["#FF1F5B","#00CD6C","#009ADE","#AF58BA","#FFC61E","#F28522","royalblue"]


fig, ax = plt.subplots(2,1)
for i,filename in enumerate(files):
    g= filename.replace(".h5","").split("_")[-1]
    cw= filename.replace(".h5","").split("_")[-2]
    X,Y,df,U,Q   = get_defect_coords(filename)
    center_y_index = Y.shape[0] // 2
    ax[0].plot(X[0, :],U[center_y_index, :],label=r"$U_e$, $g={{{}}}$, $c_w={{{}}}$".format(g,cw))

    ax[1].plot(X[0, :],df[center_y_index, :],label=r"$P$, $g={{{}}}$, $c_w={{{}}}$".format(g,cw))

ax[0].plot(X[0, :],Q[center_y_index, :],label=r"$Q$".format(g,cw))
ax[0].legend()
ax[1].legend()
plt.show()