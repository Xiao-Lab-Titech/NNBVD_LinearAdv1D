"""
IN:../PostProcessedData/xxxx.dat
OUT:./ONNX/xxxx.onnx

COMMENT:
Training a model.

"""


import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
import random
import torch.onnx


#--------------#
#  PARAMETER   #
#--------------#
fgs = 6 # short side length of figure based silver ratio (1 : square root of 2)
fmr = 0.125 # figure margin ratio
wsp = 0.2 # the amount of width reserved for space between subplots
hsp = 0.2 # the amount of height reserved for space between subplots
llw = 2 # lines linewidth
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

input_file = "./PostProcessedData/6WBVDdataset_diff5_nolog_abs.dat"
output_file = "./ONNX/tmp.onnx"
model_name = "tmp"
#dmmy_input = torch.tensor([[0, 0.166667, 0.333333, 0.5, 0.666667, 0.8, 0.9]])
dmmy_input = torch.tensor([[0, 0.166667, 0.333333, 0.5, 0.666667, 0.8, 0.9,0.123,0.234,0.534,0.2134,0.51345,0.234,0.1235,0.1324,0.513]]) # 16
#dmmy_input = torch.tensor([[0, 0.166667, 0.333333, 0.5, 0.666667, 0.8, 0.9,0.123,0.234,0.534,0.2134,0.51345,0.234,0.1235,0.1324,0.513, 0.123]]) # 17


N_stencil = 17
N_kernels = 6

batch_size = 256
epoch = 100
learning_rate = 0.001




#-----------------------#
#  FUNCTION AND CLASS   #
#-----------------------#

# 乱数シード固定
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# データローダーのサブプロセスの乱数seedが固定
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
     

class PostProcessedDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
 

# WENO判定をゆるく，THINC判定をきびしく．連続解でTHINC判定だと数値散逸が大きいため
# ↓
# false-positiveを少なくしたい
# lossが大きいほど最適化が進む
class WeightedFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, omega0=0.25, omega1=0.25, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma 
        self.omega1 = omega1
        self.omega0 = omega0
        self.reduction = reduction
    
    def forward(self, predict, target):
        pt = predict
        loss0 = -self.omega0 * pt ** self.gamma * (1-target) * torch.log(1-pt+1e-10)
        loss1 = -self.omega1 * (1-pt) ** self.gamma * target * torch.log(pt+1e-10)
        if self.reduction == 'mean':
            loss0 = torch.mean(loss0)
            loss1 = torch.mean(loss1)
            loss = loss0 + loss1
        elif self.reduction == 'sum':
            loss = torch.sum(loss0 + loss1)
        return loss
     

class WeightCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight, reduction='mean'):
        super(WeightCrossEntropyLoss, self).__init__()
        self.weight = weight 
        self.reduction = reduction

    def forward(self, predict, target):
        log_softmax_outputs = torch.log_softmax(predict, dim = 1)
        # print(f"log_sofmax:{log_softmax_outputs}")
        weight_softmax_outputs = self.weight * log_softmax_outputs
        # print(f"weight_soft:{weight_softmax_outputs}")
        loss = - torch.sum(target * weight_softmax_outputs, dim = 1)
        # print(f"loss:{loss}, target{target}")
        if self.reduction == 'mean':
            loss = torch.mean(loss)
            # print("final loss:{loss}")
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class simpleMLP(torch.nn.Module):
    def __init__(self, n_input, n_output, *n_hiddens):
        super(simpleMLP, self).__init__()

        activate = torch.nn.Sigmoid()
        self.model = torch.nn.Sequential()
        self.model.add_module("I" ,torch.nn.Linear(n_input, n_hiddens[0]))
        for i in range(len(n_hiddens) - 1):
            self.model.add_module("hidden", torch.nn.Linear(n_hiddens[i], n_hiddens[i+1]))
            self.model.add_module("batch_norm", torch.nn.BatchNorm1d(n_hiddens[i+1]))
            self.model.add_module("act", activate)
        self.model.add_module("O", torch.nn.Linear(n_hiddens[-1], n_output))
        self.model.add_module("end", torch.nn.Sigmoid())
        #self.model.add_module("softmax", torch.nn.Softmax(dim=1)) # 行単位で1にする
        for m in self.model:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1.0) # Xavier's init
                #torch.nn.init.kaiming_normal_(net.weight) # He's init



    def forward(self, x):
        y = self.model(x)
        return y 
    

# モデル訓練
def train_model(model, train_loader, test_loader):
    # Train loop ----------------------------
    model.train()  # 学習モードをオン
    train_batch_loss = []
    for data, label in train_loader: # バッチ毎に
        # GPUへの転送
        data, label = data.to(device), label.to(device)
        # 1. 勾配リセット
        optimizer.zero_grad()
        # 2. 推論
        output = model(data)
        # 3. 誤差計算
        loss = criterion(output, label)
        # 4. 誤差逆伝播
        loss.backward()
        # 5. パラメータ更新
        optimizer.step()
        # train_lossの取得
        train_batch_loss.append(loss.item())

    # Test(val) loop ----------------------------
    model.eval()  # 学習モードをオフ
    test_batch_loss = []
    with torch.no_grad():  # 勾配を計算なし
        for data, label in test_loader: # バッチ毎に
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            test_batch_loss.append(loss.item())

    return model, np.mean(train_batch_loss), np.mean(test_batch_loss)

def plot_loss(loss_train, loss_test, output_file):
    ep = np.arange(epoch)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ep, loss_train, label="train")
    ax.plot(ep, loss_test, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, epoch)
    ax.grid()
    ax.get_xaxis().set_tick_params(pad=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.clf()

def plot_confusion_matrix(cm, classes, output_file, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(output_file)
    plt.clf()
    




#--------------#
#  MAIN CODE   #
#--------------#
# Setting GPU by Pytorch
#print(os.cpu_count())
#mp.set_start_method("spawn", force=True)

print("Loading device...",end="")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#device = torch.device("mps") # for mac
print("Recognize")
#print("# of GPU: {}".format(torch.cuda.device_count()))
#print("Num_0 of GPU Name: {}".format(torch.cuda.get_device_name(torch.device("cuda:0"))))

#print(torch.cuda.is_available())

print("Setting seed...",end="")
setup_seed(1000)
print("OK")

# Load dataset
Data = np.loadtxt(input_file)
X = Data[:, 0:N_stencil]
#sign = Data[:, N_stencil].reshape(-1,1)
#label = Data[:, -1].reshape(-1,1)
label = Data[:, -1].reshape(-1).astype(int)
#label_oh = np.zeros((label.shape[0], N_kernels))
#for i in range(Data.shape[0]):
#    label_oh[i, label[i]] = 1.0
#print(label)

# Normalize dataset
#X_trans = X.transpose()
#scale = preprocessing.MinMaxScaler(feature_range=(0,1))
#X_trans_scale = scale.fit_transform(X_trans)
#X_scale = X_trans_scale.transpose()
X_scale = X # if Normalized before

# Convert to Pytorch tensor and send it to the GPU
#X_unsplit = np.hstack([X_scale, sign])
X_unsplit = np.hstack([X_scale])
X_data = torch.from_numpy(X_unsplit).float()
y_data = torch.from_numpy(label).long()

# Split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=712)
#X_train = X_train.to(device)
#X_test = X_test.to(device)
#y_train = y_train.to(device)
#y_test = y_test.to(device)


#width = 8
train_data = PostProcessedDataset(X_train, y_train)
test_data = PostProcessedDataset(X_test, y_test)
#train_data = torch.utils.data.Dataset(X_train, y_train)
#test_data= torch.utils.data.Dataset(X_test, y_test)
train_dataLoader = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
test_dataLoader  = DataLoader(test_data,  batch_size, shuffle=True, pin_memory=True)
#train_dataLoader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True, persistent_workers=True, pin_memory=True, num_workers=2) # for windows
#test_dataLoader  = DataLoader(test_data,  batch_size, shuffle=True, persistent_workers=True, pin_memory=True, num_workers=2)

#net = Net(7, 5, 5, 1).to(device) # Huangnet
net = simpleMLP(N_stencil,N_kernels,16,16,16).to(device) #
#net = Net2(3, 8, 2, 7, 8, 1).to(device) # simplenet

# optimizer = torch.optim.LBFGS(net.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# loss_fn = torch.nn.BCELoss()
# loss_fn = BCE_WITH_WEIGHT()
#criterion = WeightedFocalLoss()
criterion = torch.nn.CrossEntropyLoss()
#criterion = WeightCrossEntropyLoss(weight=1.0, reduction='mean')
train_loss = []
test_loss = []

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
#torch.nn.init.xavier_normal_(net.weight, gain=1.0) # Xavier's init
#torch.nn.init.kaiming_normal_(net.weight) # He's init

for i in range(epoch):
    net, train_l, test_l = train_model(net, train_dataLoader, test_dataLoader)
    train_loss.append(train_l)
    test_loss.append(test_l)
    # 10エポックごとにロスを表示
    if i % 10 == 0:
        print("epoch: {0}/{1}, Train loss: {2:.3f}, Test loss: {3:.3f}" \
            .format(i, epoch, train_loss[-1], test_loss[-1]))
    if i % 100 == 0:
        lr_scheduler.step()


plot_loss(train_loss, test_loss, "./loss_"+model_name+".png")

"""
y_pred = net(X_test.to(device))
for i in range(len(y_pred)):
    if (y_pred[i] < 0.5) :
        y_pred[i] = 0
    else :
        y_pred[i] = 1
"""

y_pred = torch.argmax(net(X_test.to(device)), dim=1)

cm = confusion_matrix(y_test, y_pred.detach().cpu().numpy(), labels=range(N_kernels))
print(cm)
#print(classification_report(y_test, y_pred, digits=4))
#tn, fp, fn, tp = cm.flatten()
accuracy = np.trace(cm) / np.sum(cm) * 100.0
#accuracy = (tn+tp)/(tn+fp+fn+tp)*100.0
#precision = tp/(tp+fp)*100.0
#print('Accuracy: %.4f%%, Precision: %.4f%%' % (accuracy, precision))
print('Accuracy: %.4f%%' % (accuracy))

plot_confusion_matrix(cm, range(N_kernels), "./cm_"+model_name+".png")

"""
print(classification_report(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy()))
"""

# Export onnx
net_cpu = net.cpu()
net.eval()
#dmmy_input = torch.tensor([[0, 0.166667, 0.333333, 0.5, 0.666667]])
#print(dmmy_input)
torch.onnx.export(net_cpu, dmmy_input, output_file, input_names=['input'],
                output_names=['output'], dynamic_axes= {'input':
                            {0: 'batch_size'},
                    'output':
                            {0: 'batch_size'}
                    })
