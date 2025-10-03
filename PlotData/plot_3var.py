import matplotlib.pyplot as plt
import numpy as np

fgs = 6 # short side length of figure based silver ratio (1 : square root of 2)
fmr = 0.125 # figure margin ratio
wsp = 0.2 # the amount of width reserved for space between subplots
hsp = 0.2 # the amount of height reserved for space between subplots
llw = 1 # lines linewidth
alw = 1 # axes linewidth, tick width
mks = 2 ** 8 # marker size
fts = 24 # font size
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


#case 22 
file0 = "Data/HLLC_MTBVD_order2_case0_RK2_200.dat"
file1 = "Data/HLLC_MTBVD_order2_case0_RK2_pyml_10_1_0.onnx_200.dat"
file2 = "Data/exact_sod.dat"

data0 = np.loadtxt(file0)
data1 = np.loadtxt(file1)
data2 = np.loadtxt(file2)

# print(data0)

x0 = data0[:,0]
d0 = data0[:,1]
u0 = data0[:,2]
p0 = data0[:,3]

x1 = data1[:,0]
d1 = data1[:,1]
u1 = data1[:,2]
p1 = data1[:,3]

x2 = data2[:,0]
d2 = data2[:,1]
u2 = data2[:,2]
p2 = data2[:,3]

#plt.figure()
# default figsize=(6.4,4.8)
fig=plt.figure(figsize=(9,12))
ax1=fig.add_subplot(311)
ax2=fig.add_subplot(312)
ax3=fig.add_subplot(313)



ax1.plot(x0,d0,color="none",markeredgecolor='blue',markersize=6,markeredgewidth=1,alpha=0.8,marker="o",ls="",label="MTBVD")
ax1.plot(x1,d1,color="none",markeredgecolor='green',markersize=6,markeredgewidth=1,alpha=0.8,marker="s",ls="",label="DeepMTBVD")
ax1.plot(x2,d2,color="red",label="exact")

ax2.plot(x0,u0,color="none",markeredgecolor='blue',markersize=6,markeredgewidth=1,alpha=0.8,marker="o",ls="",label="MTBVD")
ax2.plot(x1,u1,color="none",markeredgecolor='green',markersize=6,markeredgewidth=1,alpha=0.8,marker="s",ls="",label="DeepMTBVD")
ax2.plot(x2,u2,color="red",label="exact")

ax3.plot(x0,p0,color="none",markeredgecolor='blue',markersize=6,markeredgewidth=1,alpha=0.8,marker="o",ls="",label="MTBVD")
ax3.plot(x1,p1,color="none",markeredgecolor='green',markersize=6,markeredgewidth=1,alpha=0.8,marker="s",ls="",label="DeepMTBVD")
ax3.plot(x2,p2,color="red",label="exact")



ax1.set_xlabel('$x$')
ax1.set_ylabel('$\\rho$')
ax1.grid()
ax1.set_xlim(0.65,0.8)
ax1.set_ylim(0.25,0.45)
ax1.get_xaxis().set_tick_params(pad=8)

ax2.set_xlabel('$x$')
ax2.set_ylabel('$u$')
ax2.grid()
ax2.set_xlim(0.0,1.0)
ax2.set_ylim(-0.1,1.1)
ax2.get_xaxis().set_tick_params(pad=8)

ax3.set_xlabel('$x$')
ax3.set_ylabel('$p$')
ax3.grid()
ax3.set_xlim(0.0,1.0)
ax3.set_ylim(-0.1,1.1)
ax3.get_xaxis().set_tick_params(pad=8)

plt.legend(loc='upper center', bbox_to_anchor=(.5, -.5), ncol=3)
plt.tight_layout()
plt.savefig("plot_num_result_045.png")