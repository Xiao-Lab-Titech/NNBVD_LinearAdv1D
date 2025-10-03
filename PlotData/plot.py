import matplotlib.pyplot as plt
import numpy as np

fgs = 6 # short side length of figure based silver ratio (1 : square root of 2)
fmr = 0.125 # figure margin ratio
wsp = 0.2 # the amount of width reserved for space between subplots
hsp = 0.2 # the amount of height reserved for space between subplots
llw = 1 # lines linewidth
alw = 1 # axes linewidth, tick width
mks = 2 ** 8 # marker size
fts = 16 # font size
#ftf = "Times New Roman" # font.family

plt.rcParams["figure.subplot.wspace"] = wsp
plt.rcParams["figure.subplot.hspace"] = hsp
plt.rcParams["lines.linewidth"] = llw
plt.rcParams["lines.markeredgewidth"] = 0
plt.rcParams["axes.linewidth"] = alw
plt.rcParams["xtick.major.width"] = alw
plt.rcParams["ytick.major.width"] = alw
plt.rcParams["xtick.minor.width"] = alw
plt.rcParams["ytick.minor.width"] = alw
plt.rcParams["xtick.minor.visible"] = 'True'
plt.rcParams["ytick.minor.visible"] = 'True'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["font.size"] = fts
plt.rcParams["font.family"] = 'serif'
#plt.rcParams["font.serif"] = ftf
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams["legend.fontsize"] = "small"


file0 = "p1_W3TBVD.dat"
file1 = "p1_MLBasedW3TBVD.dat"
file2 = "p1_Nx200_MLBasedW3TBVD.dat"
#file2 = "exact_sod.dat"

data0 = np.loadtxt(file0)
data1 = np.loadtxt(file1)
data2 = np.loadtxt(file2)


x0 = data0[:,0]
d0 = data0[:,1]
d1 = data1[:,1]
d2 = data0[:,7]
d3 = data2[:,1]


#plt.figure()
# default figsize=(6.4,4.8)
fig=plt.figure(figsize=(6.4,4.8))
ax=fig.add_subplot(111)


ax.plot(x0,d0,color="none",markeredgecolor='blue',markersize=6,markeredgewidth=1,alpha=0.8,marker="o",ls="",label="WTBVD")
ax.plot(x0,d1,color="none",markeredgecolor='green',markersize=6,markeredgewidth=1,alpha=0.8,marker="s",ls="",label="WT-Hybrid")
ax.plot(x0,d3,color="none",markeredgecolor='orange',markersize=6,markeredgewidth=1,alpha=0.8,marker="s",ls="",label="WT-Hybrid mod.")
ax.plot(x0,d2,color="red",label="Exact")



ax.set_xlabel('$x$')
ax.set_ylabel('$\\rho$')
ax.grid()
ax.set_xlim(0.0,1.0)
ax.set_ylim(0.0,1.4)
ax.get_xaxis().set_tick_params(pad=8)

plt.legend()
plt.tight_layout()
plt.savefig("plot.png")
plt.show()
