import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker

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


#file0 = "p1_W3TBVD.dat"
#file1 = "p1_MLBasedW3TBVD.dat"
#file2 = "exact_sod.dat"

#data0 = np.loadtxt(file0)
#data1 = np.loadtxt(file1)
#data2 = np.loadtxt(file2)


x0 = [200, 400, 800, 1600, 3200, 6400, 12800]
d0 = [2.252e-05
,2.604e-06
,3.026e-07
,3.563e-08
,4.316e-09
,4.439e-10
,5.579e-11
] #5.756, 5.673, 5.711, 6.250
d1 = [3.809e-05
,4.527e-06
,5.141e-07
,5.887e-08
,7.813e-09
,1.087e-09
,1.575e-10
] #5.756, 5.673, 5.711, 6.250
d2 = [2.231e-05
,2.616e-06
,3.109e-07
,3.575e-08
,4.240e-09
,4.438e-10
,5.579e-11
]
d3 = [6.135e-05
,8.594e-06
,1.307e-06
,2.046e-07
,3.637e-08
,6.427e-09
,1.293e-09
]


#plt.figure()
# default figsize=(6.4,4.8)
fig=plt.figure(figsize=(6.4,4.8))
ax=fig.add_subplot(111)


#ax.plot(x0,d0,color="none",markeredgecolor='blue',markersize=6,markeredgewidth=1,alpha=0.8,marker="o",ls="",label="WTBVD")
ax.plot(x0,d0,color="blue",markerfacecolor="white",markeredgecolor='blue',markersize=6,markeredgewidth=1,alpha=0.8,marker="s",ls="-",label="BVD org.")
ax.plot(x0,d1,color="green",markerfacecolor="white",markeredgecolor='green',markersize=6,markeredgewidth=1,alpha=0.8,marker="s",ls="-",label="S BVD 1")
ax.plot(x0,d2,color="red",markerfacecolor="white",markeredgecolor='red',markersize=6,markeredgewidth=1,alpha=0.8,marker="s",ls="-",label="S BVD 2")
ax.plot(x0,d3,color="black",markerfacecolor="white",markeredgecolor='black',markersize=6,markeredgewidth=1,alpha=0.8,marker="s",ls="-",label="S BVD 3")


ax.set_xlabel('$N_x$')
ax.set_ylabel('Total time')
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid()
#ax.set_xlim(1500,18000)
#ax.set_ylim(1.0,1.4)
#ax.get_xaxis().set_tick_params(pad=8)
ax.set_xticks([2e3,1e4], minor=True)
#ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(locs=[2e3,3e3,4e3,5e3]))
#ax.tick_params(which="minor",length=3)

plt.legend()
plt.tight_layout()
plt.savefig("ct_diff_tbv_L2_8.png")