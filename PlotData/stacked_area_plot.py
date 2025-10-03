import matplotlib.pyplot as plt
import numpy as np
import math
import glob

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

# 補間関数の種類数
num_kernels = 2
end_ts = 3999
substep = 3
ts_period = 50
#file = "p2_Jiang&Shu_Nx200_nW5BVD_Roe_RK3"
file = "p2_Jiang&Shu_Nx200_MLBasednWBVD_Roe_RK3"

num_timesteps = math.ceil(end_ts/ts_period) + 1

# 割合を保存する配列 (補間関数数, タイムステップ数)
ratios = np.zeros((num_kernels, num_timesteps*substep))
filepaths = glob.glob("./"+file+"/*.dat")

for i in range(len(filepaths)):
    #print(filepaths[i])
    data = np.loadtxt(filepaths[i])
    col0 = data[0, 0].astype(int)
    col6 = data[:, 3].astype(int)
    counts = np.bincount(col6, minlength=num_kernels)
    ratios[:, math.ceil(col0/ts_period)*3+int(filepaths[i][-5])-1] = counts/counts.sum()

t = np.linspace(0, end_ts, num_timesteps*substep)



#labels = ["WENO-A","WENO-JS","WENO-NIP","WENO-Z","WENO-ZA","WENO-Z+","WENO-$\eta$"]
#labels = ["WENO-JS","WENO-Z","WENO-$\eta$","WENO-Z+","WENO-ZA","WENO-A","WENO-NIP"]
labels = ["JS","Z"]
#labels = ["WENO-JS","WENO-Z","WENO-$\eta$","WENO-Z+","WENO-ZA","WENO-A"]
#labels.sort()
colors = plt.cm.tab10(np.linspace(0, 1, num_kernels))  # 自動で色を割り当て

#plt.figure()
# default figsize=(6.4,4.8)=(4,3)
fig=plt.figure(figsize=(7.2,4.8))
ax=fig.add_subplot(111)

ax.stackplot(t, ratios, labels=labels)


ax.set_xlabel('Time step')
ax.set_ylabel('Ratio selected by BVD algorithm')
ax.get_xaxis().set_tick_params(pad=8)
ax.set_xlim(0, end_ts)
ax.set_ylim(0.0, 1.0)
ax.legend(loc='center left', bbox_to_anchor=(1, .5))
#plt.legend(loc='center left', bbox_to_anchor=(1, .5), ncol=3)


plt.tight_layout()
plt.savefig("saplot_"+file+".png")
plt.show()