import matplotlib.pyplot as plt
import numpy as np
import h5py






figsh, ax = plt.subplots(figsize=(4, 4))
dataSH = np.loadtxt("./out/csv/smooth/SG_energy.csv",delimiter="\t")

ax.plot(dataSH[:,0],dataSH[:,1],lw=0.9,color="royalblue")
# ax.scatter(dataSH[:,0],dataSH[:,1],marker="x",color="royalblue",alpha=0.2)
# ax.set_ylim(-1e6,0)
# ax.set_yscale("symlog")
plt.show()


figuq, ax2 = plt.subplots(figsize=(4, 4))

datauq = np.loadtxt("./out/csv/smooth/errors1.csv",delimiter="\t")
l2 = np.sqrt(datauq[:,1]**2 +datauq[:,2]**2 +datauq[:,3]**2 +datauq[:,4]**2)
gradE = np.gradient(l2)/0.1
# ax2.plot(datauq[:,0],gradE,color="black")
ax2.plot(datauq[:,0],l2,color="black")
ax2.plot(datauq[:,0],datauq[:,1],lw=0.9,ls="dashed",color="green",label="11")
ax2.plot(datauq[:,0],datauq[:,2],lw=0.9,ls="dashed",color="red",label="12")
ax2.plot(datauq[:,0],datauq[:,3],lw=0.9,ls="dashed",color="cyan",label="21")
ax2.plot(datauq[:,0],datauq[:,4],lw=0.9,ls="dashed",color="purple",label="22")

ax2.set_yscale("log")
plt.show()