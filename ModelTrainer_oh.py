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
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
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

#input_file = {"./PostProcessedData/6WBVDdataset_nrm1_d15_3000_1000_4.dat"}
input_file = {"./PostProcessedData/test.dat"}
output_file = "./ONNX/test.onnx"
model_name = "test"
#dmmy_input = torch.tensor([[0, 0.166667, 0.333333, 0.5, 0.666667, 0.8, 0.9]])
dmmy_input = torch.tensor([[0, 0.166667, 0.333333, 0.5, 0.666667, 0.8, 0.9,0.123,0.234,0.534,0.2134,0.51345,0.234,0.1235,0.1324,0.513]]) # 16
#dmmy_input = torch.tensor([[0, 0.166667, 0.333333, 0.5, 0.666667, 0.8, 0.9,0.123,0.234,0.534,0.2134,0.51345,0.234,0.1235,0.1324,0.513, 0.123]]) # 17


N_stencil = 16
N_kernels = 6

batch_size = 256
epoch = 500
learning_rate = 0.0001




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
    def __init__(self, gamma=2.0, omega0=2.0, omega1=0.25, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma 
        self.omega1 = omega1
        self.omega0 = omega0
        self.reduction = reduction
    
    def forward(self, predict, target):
        pt = predict
        pt_sigmoid = torch.sigmoid(pt)
        loss0 = -self.omega0 * pt_sigmoid ** self.gamma * (1-target) * torch.log(1-pt_sigmoid+1e-10)
        loss1 = -self.omega1 * (1-pt_sigmoid) ** self.gamma * target * torch.log(pt_sigmoid+1e-10)
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

class WeightedBCELoss(torch.nn.Module):
    def __init__(self, pos_weight=1.5, neg_weight=2.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, predict, target):
        pt_sigmoid = torch.sigmoid(predict)
        loss_pos = - target * torch.log(pt_sigmoid+1e-10)
        loss_neg = - (1-target) * torch.log(1-pt_sigmoid+1e-10)
        return (self.pos_weight * loss_pos + self.neg_weight * loss_neg).mean()


class simpleMLP(torch.nn.Module):
    def __init__(self, n_input, n_output, *n_hiddens):
        super(simpleMLP, self).__init__()

        activate = torch.nn.ReLU()
        self.model = torch.nn.Sequential()
        self.model.add_module("I" ,torch.nn.Linear(n_input, n_hiddens[0]))
        for i in range(len(n_hiddens) - 1):
            self.model.add_module(f"hidden{i}", torch.nn.Linear(n_hiddens[i], n_hiddens[i+1]))
            self.model.add_module(f"batch_norm{i}", torch.nn.BatchNorm1d(n_hiddens[i+1]))
            self.model.add_module(f"act{i}", activate)
        self.model.add_module("O", torch.nn.Linear(n_hiddens[-1], n_output))
        #self.model.add_module("end", torch.nn.Sigmoid())
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
    
def plot_ROC(classes, output_file, y_test, y_pred_probs):
    # 予測確率と実際のラベルを用意
    y_score = y_pred_probs.detach().cpu().numpy() if hasattr(y_pred_probs, 'cpu') else y_pred_probs.detach().numpy()
    y_true = y_test.cpu().numpy() if hasattr(y_test, 'cpu') else y_test.numpy()

    # 各クラスごとにROC曲線を計算
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # ROC曲線をプロット
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(classes):
        ax.plot(fpr[i], tpr[i], label=f'Class{i} (AUC:{roc_auc[i]:.2f})')

    ax.plot([0, 1], [0, 1], 'k--')  # 対角線
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.get_xaxis().set_tick_params(pad=8)
    #ax.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.clf()




#--------------#
#  MAIN CODE   #
#--------------#
# Setting GPU by Pytorch
#print(os.cpu_count())
#mp.set_start_method("spawn", force=True)

print("Loading device...",end="")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#device = torch.device("mps") # for mac
print("Recognize")
print("# of GPU: {}".format(torch.cuda.device_count()))
print("Num_0 of GPU Name: {}".format(torch.cuda.get_device_name(torch.device("cuda:0"))))

#print(torch.cuda.is_available())

print("Setting seed...",end="")
setup_seed(1000)
print("OK")

# Load dataset
Data = np.empty((0, N_stencil + N_kernels))
for file in input_file:
    print(f"Load {file}")
    Data_tmp = np.loadtxt(file)
    Data = np.vstack([Data, Data_tmp]) if 'Data' in locals() else Data_tmp

X = Data[:, 0:N_stencil]
label = Data[:, -N_kernels:] # one-hot label

X_scale = X # if Normalized before

# Convert to Pytorch tensor and send it to the GPU
X_unsplit = np.hstack([X_scale])
X_data = torch.from_numpy(X_unsplit).float()
y_data = torch.from_numpy(label).float()

print(y_data.shape)
# Split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=712)

#print(y_test.shape)

train_data = PostProcessedDataset(X_train, y_train)
test_data = PostProcessedDataset(X_test, y_test)
train_dataLoader = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True, num_workers=2, worker_init_fn=worker_init_fn)
test_dataLoader  = DataLoader(test_data,  batch_size, shuffle=True, pin_memory=True, num_workers=2, worker_init_fn=worker_init_fn)
#train_dataLoader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True, persistent_workers=True, pin_memory=True, num_workers=2) # for windows
#test_dataLoader  = DataLoader(test_data,  batch_size, shuffle=True, persistent_workers=True, pin_memory=True, num_workers=2)

net = simpleMLP(N_stencil,N_kernels,16,16,16).to(device) #

#criterion = torch.nn.BCEWithLogitsLoss()
criterion = WeightedBCELoss()
#criterion = WeightedFocalLoss()
train_loss = []
test_loss = []

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

for i in range(epoch):
    net, train_l, test_l = train_model(net, train_dataLoader, test_dataLoader)
    train_loss.append(train_l)
    test_loss.append(test_l)
    if i % 10 == 0: # show loss by 10 epochs
        print("epoch: {0}/{1}, Train loss: {2:.3f}, Test loss: {3:.3f}" \
            .format(i, epoch, train_loss[-1], test_loss[-1]))
    if i % 100 == 0:
        lr_scheduler.step()


plot_loss(train_loss, test_loss, "./loss_"+model_name+".png")



y_pred_logits = net(X_test.to(device))
y_pred_probs = torch.sigmoid(y_pred_logits)
#y_pred_probs = y_pred_logits
#print(y_pred_probs)
y_pred = (y_pred_probs > 0.7).int()  # 閾値で判定
# 各サンプルごとに最大値のインデックスだけ1、それ以外は0にする
y_pred_max = torch.zeros_like(y_pred_probs)
max_indices = torch.argmax(y_pred_probs, dim=1)
y_pred_max[torch.arange(y_pred_probs.size(0)), max_indices] = 1

y_test_int = (y_test > 0.7).int()  # 閾値で判定
#print(y_pred_max)
#print(y_test)
print(y_pred.shape, y_pred_max.shape, y_test_int.shape)


acc_g = 0
for i in range(len(y_pred)):
    tm = 0
    for j in range(N_kernels):
        if (y_pred_max[i, j] == 1 and y_test_int[i, j] == 1):
            tm += 1
    if (tm >= 1):
        acc_g += 1


print(f"acc_g: {acc_g:.4f}, Size: {len(y_pred_max)}, acc_g(%): {acc_g/len(y_pred_max)*100:.4f}%")

y_test_np = y_test_int.cpu().numpy().astype(int) if hasattr(y_test_int, 'cpu') else y_test_int.numpy().astype(int)
y_pred_np = y_pred.cpu().numpy().astype(int) if hasattr(y_pred, 'cpu') else y_pred.numpy().astype(int)
print("y_test_np shape:", y_test_np.shape)
print("y_pred_np shape:", y_pred_np.shape)
#print("unique labels in y_test_np:", np.unique(y_test_np))

plot_ROC(N_kernels, "./ROC_"+model_name+".png", y_test_int, y_pred_probs)

print(classification_report(
    y_test_np, y_pred_np,
    digits=4,
    target_names=[f"class{i}" for i in range(N_kernels)],
    zero_division=0
))

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

