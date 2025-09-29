



import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.ndimage as ndimage
def delta(Z,w):
    return (1/(2*np.pi*w**2))*np.exp(-(Z**2)/(2*w**2))


a0 = 4*np.pi/np.sqrt(3)
xmin, xmax = 150, 320
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
            for t in range(len(sorted_timesteps)):
                # t=60
                alphaFE = np.abs(f["Function/alpha/"+sorted_timesteps[t]])
                alphaFE= alphaFE[:,2]
                alpha = alphaFE[mapping].reshape(ny, nx)
                # plt.contourf(X,Y,alpha,levels=200,cmap="seismic")
                # plt.show()

                # exit()

                filtered_F = ndimage.maximum_filter(alpha, size=5)
                peaks = np.where(alpha == filtered_F)

                # Extract positions and values
                peak_x = X[peaks]
                peak_y = Y[peaks]
                peak_values = alpha[peaks]

                # Sort peaks by intensity (descending)
                sorted_indices = np.argsort(peak_values)[::-1]
                peak_x = peak_x[sorted_indices]
                peak_y = peak_y[sorted_indices]
                peak_values = peak_values[sorted_indices]


                # Define threshold and bounds
                threshold = 0.5 * np.max(alpha)
                mask_value = peak_values > threshold
                mask_bounds = (
                    (peak_x >= xmin) & (peak_x <= xmax) &
                    (peak_y >= ymin) & (peak_y <= ymax)
                )
                final_mask = mask_value & mask_bounds

                peak_x = peak_x[final_mask]
                peak_y = peak_y[final_mask]
                peak_values = peak_values[final_mask]


                if len(peak_values) >= 2:
                    List_x = np.array([peak_x[-1],peak_x[-2]])
                    List_y = np.array([peak_y[-1],peak_y[-2]])
                    sorted_x = np.argsort(List_x)
                    List_x= List_x[sorted_x]
                    List_y= List_y[sorted_x]
                    x1, y1 = List_x[-1], List_y[-1]
                    x2, y2 = List_x[-2], List_y[-2]
                    nodes=[(x1,y1),(x2,y2)]

                    r=a0*0.8
                    for _ in range(2):
                        x0,y0 = nodes[_]
                        mask = (X - x0)**2 + (Y - y0)**2 <= r**2
                        alpha_masked = alpha[mask]
                        X_masked = X[mask]
                        Y_masked = Y[mask]
                        total_mass = np.sum(alpha_masked) * dA
                        if total_mass == 0:
                            if _==0:
                                defect1_loc.append([np.nan, np.nan])
                            elif _==1:
                                defect2_loc.append([np.nan, np.nan])
                            continue
                        
                        x_com = np.sum(alpha_masked * X_masked) * dA / total_mass
                        y_com = np.sum(alpha_masked * Y_masked) * dA / total_mass
                        if _==0:
                            defect1_loc.append(np.array([x_com, y_com]))
                        elif _==1:
                            defect2_loc.append(np.array([x_com, y_com]))
                    d=np.sqrt((defect1_loc[-1][0]-defect2_loc[-1][0])**2+(defect1_loc[-1][1]-defect2_loc[-1][1])**2)
                    ts.append(float(sorted_timesteps[t].replace("_",".")))

                else:
                    if len(defect1_loc)>0:
                        x_com, y_com = defect1_loc[-1][0], defect1_loc[-1][1]
                    else:
                        x_com, y_com = 0,0
                    d=0
                    defect1_loc.append(np.array([x_com, y_com]))
                    defect2_loc.append(np.array([x_com, y_com]))
                    ts.append(float(sorted_timesteps[t].replace("_",".")))

                if False:
                    break
    return ts,        np.array(defect1_loc)    ,np.array(defect2_loc)    




files=[
# "out/dipole/Pen0.1_0.1_0.08.h5",
# "out/dipole/Pen0.1_0.1_0.05.h5",
"out/dipole/Pen0.1_0_0.h5",
"out/dipole/Pen0.1_0.01_50.h5",
"out/dipole/Pen0.1_0.01_20.h5",

]
colors=["#FF1F5B","#00CD6C","#009ADE","#AF58BA","#FFC61E","#F28522","royalblue"]
for i,filename in enumerate(files):
    ts,        defect1_loc    ,defect2_loc   = get_defect_coords(filename)

    print(ts)
    g = filename.replace(".h5","").split("_")[-1]
    # print(dt,cw,r,g )
    # cw=0
    # g=0
    plt.axhline(0.5*(defect1_loc[0,0]+defect2_loc[0,0]))
    if filename=="out/trash/ShearCompFFT0.1_0_.h5":
        plt.plot(ts,defect1_loc[:,0],color="black",label="cw=0")
        plt.plot(ts,defect2_loc[:,0],color="black")
    else:
        plt.plot(ts,defect1_loc[:,0],color=colors[i],label="g="+g)
        plt.plot(ts,defect2_loc[:,0],color=colors[i])
plt.legend()
plt.show()