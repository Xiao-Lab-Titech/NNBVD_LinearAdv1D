"""
IN:../PreProcessedData/xxx/
OUT:xxxdataset.dat

COMMENT:
Processing raw data into dataset.
The dataset include training and test data.

"""

import torch
import numpy as np
import os
from collections import namedtuple
import random



#--------------#
#  PARAMETER   #
#--------------#

input_files = "./PreProcessedData/6WBVD-d15_4/" # raw data
#input_files = "./PreProcessedData/nW5BVD_rand/" # raw data
#input_files = "./PreProcessedData/tmp/" # raw data
output_file = "./PostProcessedData/6WBVD_test_true_4.dat" # training data
N_kernels = 6 # Number of candidate func.
N_stc = 7 # Number of stencil
N_each_cases = 500
batch_size = 1000
ratio_of_extraction = 0.5
threshold_of_likelihood = 0.5

del_piecewise_const = False
del_not_monotonic = False
del_not_convex = True


input_stencil7 = False
input_diff5 = False
input_diff5_nolog = False
input_diff5_2_nolog = False
input_diff5_nolog_abs = False
input_abs_diff5 = False
input_diff5_nrm1 = True


is_normed_by_stencil = False
is_normed_by_computational_field = False


#--------------#
#   FUNCTION   #
#--------------#

# Set a random seed
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# Load raw data
def load_data(file_par_path):
    file_list = os.listdir(file_par_path)
    lines = []
    for file_path in file_list:
        with open(os.path.join(file_par_path, file_path), "r") as f:
            lines += f.readlines()
    return lines

# Return a 2d-list with duplicate data removed
def transform_2dlist(lines):
    data_set = set(lines)
    data_list = list(map(lambda x: x.replace('\n', '').split('\t'), data_set))
    for idx, l in enumerate(data_list):
        l = [item for item in l if item != '']
        data_list[idx] = list(map(float, l))
    return data_list

def remove_duplicates(arr):
    # arr: numpy array
    # Convert each row to tuple for set uniqueness
    seen = set()
    unique_rows = []
    for row in arr:
        t = tuple(row)
        if t not in seen:
            seen.add(t)
            unique_rows.append(row)
    return np.array(unique_rows)

def load_data_in_batches(file_par_path, batch_size):
    file_list = os.listdir(file_par_path)
    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i:i+batch_size]
        lines = []
        for file_path in batch_files:
            with open(os.path.join(file_par_path, file_path), "r") as f:
                lines += f.readlines()
        yield lines


#--------------#
#  MAIN CODE   #
#--------------#

# Setting device
print("Loading device...",end="")
device = torch.device("cpu") # for windows or ubuntu
#device = torch.device("mps") # for mac
print("Recognize")

print("Setting seed...",end="")
setup_seed(1234)
print("OK")

print(f"Loading raw data({input_files})...",end="")

all_x_print_list = []

for lines in load_data_in_batches(input_files, batch_size):
    data_list = transform_2dlist(lines)
    #lengths = [len(row) for row in data_list]
    #print("各行の長さ:", set(lengths))

    Data = namedtuple("Data", ["data", "Mm", "target"])
    data = Data(np.array(data_list)[:, 0:N_stc+1], 
                np.array(data_list)[:, N_stc+1:-N_kernels],
                np.array(data_list)[:, -N_kernels:])
    X = data.data
    Mm = data.Mm
    y = data.target

    print("{} raw data loaded".format(len(y)))

    counts = np.zeros(N_kernels)
    for col in range(y.shape[1]):
        counts[col] = np.sum(y[:, col] >= threshold_of_likelihood)
    for i in range(counts.shape[0]):
        print(f"case{i}:{int(counts[i])} ",end="")
    print("")


    print("\n< Processing dataset >")
    #x1 = X[:, 0:5] # for Huang's data
    x1 = X[:,1:N_stc+1]
    x1_f = [] # feature value
    #print(sign[0,:])

    print("1) Preprocessing...",end="")
    idx_del = []
    idx_cases = []


    is_monotonic = lambda q0,q1,q2: (q1-q0)*(q2-q1)>0
    is_convex = lambda q0,q1,q2: abs(q0-2.0*q1+q2)>1.e-5
    for i in range(x1.shape[0]):
        M = np.max(x1[i])
        m = np.min(x1[i])

        # delete stencils which are piecewise constant.
        if (del_piecewise_const):
            if (M - m) < 1.e-15: # Do NOT make threshold too small.
                idx_del.append(i)
        
        # delete stencils which are NOT 3-cell monotonic
        if (del_not_monotonic):
            for j in range(x1.shape[1]-2):
                if (not is_monotonic(x1[i, j], x1[i, j+1], x1[i, j+2])):
                    idx_del.append(i)
                    break

        if (del_not_convex):
            n_cp = 0
            for j in range(x1.shape[1]-2):
                #M = np.max([x1[i, j], x1[i, j+1], x1[i, j+2]])
                #m = np.min([x1[i, j], x1[i, j+1], x1[i, j+2]])
                if (not is_monotonic(x1[i, j], x1[i, j+1], x1[i, j+2])):
                    n_cp += 1
            if ((n_cp == 1 and not is_convex(x1[i, 0], x1[i, x1.shape[1]//2], x1[i, -1])) or n_cp >= 2):
                idx_del.append(i)

    if (input_stencil7):
        for i in range(x1.shape[0]):
            x1_f.append(x1[i, :])


    if (input_diff5):
        for i in range(x1.shape[0]):
            qm3,qm2,qm1,q0,qp1,qp2,qp3 = x1[i, :]
            diffs = np.zeros(16)
            diffs[0] = qp1-q0
            diffs[1] = q0-qm1

            diffs[2] = (q0-qm2)*0.5
            diffs[3] = (qp1-qm1)*0.5
            diffs[4] = (qp2-q0)*0.5

            diffs[5] = -(3.0*qm1 - 4.0*q0 + qp1)*0.5
            diffs[6] = -(3.0*q0 - 4.0*qp1 + qp2)*0.5
            diffs[7] = -(3.0*qp1 - 4.0*qp2 + qp3)*0.5

            diffs[8] = (qm3 - 4.0*qm2 + 3.0*qm1)*0.5
            diffs[9] = (qm2 - 4.0*qm1 + 3.0*q0)*0.5
            diffs[10] = (qm1 - 4.0*q0 + 3.0*qp1)*0.5

            diffs[11] = qm3 - 2.0*qm2 + qm1
            diffs[12] = qm2 - 2.0*qm1 + q0
            diffs[13] = qm1 - 2.0*q0 + qp1
            diffs[14] = q0 - 2.0*qp1 + qp2
            diffs[15] = qp1 - 2.0*qp2 + qp3

            diff_max = np.max(np.concatenate([np.abs(diffs[:2]), np.array([1.e-15])]))
            log_diff = np.log10(diff_max)
            if (diff_max > 1.e-15):
                x1_f.append(np.concatenate([diffs/diff_max, np.array([log_diff])]))
            else:
                diffs = np.zeros(16)
                x1_f.append(np.concatenate([diffs/diff_max, np.array([log_diff])]))
                idx_del.append(i)

    if (input_diff5_nolog):
        for i in range(x1.shape[0]):
            qm3,qm2,qm1,q0,qp1,qp2,qp3 = x1[i, :]
            diffs = np.zeros(16)
            diffs[0] = qp1-q0
            diffs[1] = q0-qm1

            diffs[2] = (q0-qm2)*0.5
            diffs[3] = (qp1-qm1)*0.5
            diffs[4] = (qp2-q0)*0.5

            diffs[5] = -(3.0*qm1 - 4.0*q0 + qp1)*0.5
            diffs[6] = -(3.0*q0 - 4.0*qp1 + qp2)*0.5
            diffs[7] = -(3.0*qp1 - 4.0*qp2 + qp3)*0.5

            diffs[8] = (qm3 - 4.0*qm2 + 3.0*qm1)*0.5
            diffs[9] = (qm2 - 4.0*qm1 + 3.0*q0)*0.5
            diffs[10] = (qm1 - 4.0*q0 + 3.0*qp1)*0.5

            diffs[11] = qm3 - 2.0*qm2 + qm1
            diffs[12] = qm2 - 2.0*qm1 + q0
            diffs[13] = qm1 - 2.0*q0 + qp1
            diffs[14] = q0 - 2.0*qp1 + qp2
            diffs[15] = qp1 - 2.0*qp2 + qp3

            diff_max = np.max(np.concatenate([np.abs(diffs[:2]), np.array([1.e-15])]))
            
            if (diff_max > 1.e-15):
                x1_f.append(np.concatenate([diffs/diff_max]))
            else:
                diffs = np.zeros(16)
                x1_f.append(np.concatenate([diffs/diff_max]))
                idx_del.append(i)

    if (input_diff5_2_nolog):
        for i in range(x1.shape[0]):
            qm3,qm2,qm1,q0,qp1,qp2,qp3 = x1[i, :]
            diffs = np.zeros(10)
            diffs[0] = qp1-q0
            diffs[1] = q0-qm1

            diffs[2] = (q0-qm2)*0.5
            diffs[3] = (qp1-qm1)*0.5
            diffs[4] = (qp2-q0)*0.5

            diffs[5] = qm3 - 2.0*qm2 + qm1
            diffs[6] = qm2 - 2.0*qm1 + q0
            diffs[7] = qm1 - 2.0*q0 + qp1
            diffs[8] = q0 - 2.0*qp1 + qp2
            diffs[9] = qp1 - 2.0*qp2 + qp3

            diff_max = np.max(np.concatenate([np.abs(diffs[:2]), np.array([1.e-15])]))
            
            if (diff_max > 1.e-15):
                x1_f.append(np.concatenate([diffs/diff_max]))
            else:
                diffs = np.zeros(10)
                x1_f.append(np.concatenate([diffs/diff_max]))
                idx_del.append(i)

    if (input_diff5_nolog_abs):
        for i in range(x1.shape[0]):
            qm3,qm2,qm1,q0,qp1,qp2,qp3 = x1[i, :]
            diffs = np.zeros(16)
            diffs[0] = qp1-q0
            diffs[1] = q0-qm1

            diffs[2] = (q0-qm2)*0.5
            diffs[3] = (qp1-qm1)*0.5
            diffs[4] = (qp2-q0)*0.5

            diffs[5] = -(3.0*qm1 - 4.0*q0 + qp1)*0.5
            diffs[6] = -(3.0*q0 - 4.0*qp1 + qp2)*0.5
            diffs[7] = -(3.0*qp1 - 4.0*qp2 + qp3)*0.5

            diffs[8] = (qm3 - 4.0*qm2 + 3.0*qm1)*0.5
            diffs[9] = (qm2 - 4.0*qm1 + 3.0*q0)*0.5
            diffs[10] = (qm1 - 4.0*q0 + 3.0*qp1)*0.5

            diffs[11] = qm3 - 2.0*qm2 + qm1
            diffs[12] = qm2 - 2.0*qm1 + q0
            diffs[13] = qm1 - 2.0*q0 + qp1
            diffs[14] = q0 - 2.0*qp1 + qp2
            diffs[15] = qp1 - 2.0*qp2 + qp3

            diff_max = np.max(np.concatenate([np.abs(diffs[:2]), np.array([1.e-15])]))
            
            if (diff_max > 1.e-15):
                x1_f.append(np.concatenate([np.abs(diffs)/diff_max]))
            else:
                diffs = np.zeros(16)
                x1_f.append(np.concatenate([np.abs(diffs)/diff_max]))
                idx_del.append(i)


    if (input_abs_diff5):
        for i in range(x1.shape[0]):
            qm3,qm2,qm1,q0,qp1,qp2,qp3 = x1[i, :]
            diffs = np.zeros(16)
            diffs[0] = qp1-q0
            diffs[1] = q0-qm1

            diffs[2] = (q0-qm2)*0.5
            diffs[3] = (qp1-qm1)*0.5
            diffs[4] = (qp2-q0)*0.5

            diffs[5] = -(3.0*qm1 - 4.0*q0 + qp1)*0.5
            diffs[6] = -(3.0*q0 - 4.0*qp1 + qp2)*0.5
            diffs[7] = -(3.0*qp1 - 4.0*qp2 + qp3)*0.5

            diffs[8] = (qm3 - 4.0*qm2 + 3.0*qm1)*0.5
            diffs[9] = (qm2 - 4.0*qm1 + 3.0*q0)*0.5
            diffs[10] = (qm1 - 4.0*q0 + 3.0*qp1)*0.5

            diffs[11] = qm3 - 2.0*qm2 + qm1
            diffs[12] = qm2 - 2.0*qm1 + q0
            diffs[13] = qm1 - 2.0*q0 + qp1
            diffs[14] = q0 - 2.0*qp1 + qp2
            diffs[15] = qp1 - 2.0*qp2 + qp3

            diff_max = np.max(np.concatenate([np.abs(diffs[:2]), np.array([1.e-15])]))
            log_diff = np.log10(diff_max)
            #x1_f.append(np.concatenate([np.abs(diffs)/diff_max, np.array([log_diff])]))
            if (diff_max > 1.e-15):
                x1_f.append(np.concatenate([np.abs(diffs)/diff_max, np.array([log_diff])]))
            else:
                diffs = np.zeros(16)
                x1_f.append(np.concatenate([np.abs(diffs)/diff_max, np.array([log_diff])]))
                idx_del.append(i)
            
            


    if (input_diff5_nrm1):
        for i in range(x1.shape[0]):
            qm3,qm2,qm1,q0,qp1,qp2,qp3 = x1[i, :]
            M = Mm[i, 0]
            m = Mm[i, 1]
            M_f1 = Mm[i, 2]
            m_f1 = Mm[i, 3]
            M_f2 = Mm[i, 4]
            m_f2 = Mm[i, 5]
            M_c2 = Mm[i, 6]
            m_c2 = Mm[i, 7]
            M_b2 = Mm[i, 8]
            m_b2 = Mm[i, 9]
            M_c2_2 = Mm[i, 10]
            m_c2_2 = Mm[i, 11]

            M_abs = abs(M) if (abs(M) > abs(m)) else abs(m)
            M_f1_abs = abs(M_f1) if (abs(M_f1) > abs(m_f1)) else abs(m_f1)
            M_f2_abs = abs(M_f2) if (abs(M_f2) > abs(m_f2)) else abs(m_f2)
            M_c2_abs = abs(M_c2) if (abs(M_c2) > abs(m_c2)) else abs(m_c2)
            M_b2_abs = abs(M_b2) if (abs(M_b2) > abs(m_b2)) else abs(m_b2)
            M_c2_2_abs = abs(M_c2_2) if (abs(M_c2_2) > abs(m_c2_2)) else abs(m_c2_2)
            
            input = np.zeros(16)
            input[0] = abs(qp1-q0)
            input[1] = abs(q0-qm1)

            input[2] = abs(q0-qm2)*0.5/M_c2_abs # i-1/2L
            input[3] = abs(qp1-qm1)*0.5/M_c2_abs # i+1/2L, i-1/2R
            input[4] = abs(qp2-q0)*0.5/M_c2_abs # i+1/2R

            input[5] = abs(-(3.0*qm1 - 4.0*q0 + qp1))*0.5/M_f2_abs
            input[6] = abs(-(3.0*q0 - 4.0*qp1 + qp2))*0.5/M_f2_abs
            input[7] = abs(-(3.0*qp1 - 4.0*qp2 + qp3))*0.5/M_f2_abs

            #diffs[5] = -(3.0*qm1 - 4.0*qm2 + qm3)*0.5
            #diffs[6] = -(3.0*q0 - 4.0*qm1 + qm2)*0.5
            #diffs[7] = -(3.0*q0 - 4.0*qp1 + qp2)*0.5
            #diffs[7] = -(3.0*qp1 - 4.0*qp2 + qp3)*0.5

            input[8] = abs(qm3 - 4.0*qm2 + 3.0*qm1)*0.5/M_b2_abs
            input[9] = abs(qm2 - 4.0*qm1 + 3.0*q0)*0.5/M_b2_abs
            input[10] = abs(qm1 - 4.0*q0 + 3.0*qp1)*0.5/M_b2_abs

            #diffs[8] = (qp1 - 4.0*q0 + 3.0*qm1)*0.5
            #diffs[9] = (qp2 - 4.0*qp1 + 3.0*q0)*0.5
            #diffs[10] = (qm2 - 4.0*qm1 + 3.0*q0)*0.5
            #diffs[10] = (qm1 - 4.0*q0 + 3.0*qp1)*0.5

            input[11] = abs(qm3 - 2.0*qm2 + qm1)/M_c2_2_abs # i-1/2L
            input[12] = abs(qm2 - 2.0*qm1 + q0)/M_c2_2_abs
            input[13] = abs(qm1 - 2.0*q0 + qp1)/M_c2_2_abs
            input[14] = abs(q0 - 2.0*qp1 + qp2)/M_c2_2_abs
            input[15] = abs(qp1 - 2.0*qp2 + qp3)/M_c2_2_abs # i+1/2R

            #diff_max = np.max(np.concatenate([np.abs(input[:2])]))
            #log_diff = np.tanh(diff_max)
            input[:2] /= M_f1_abs
            
            #x1_f.append(np.concatenate([np.abs(input), np.array([log_diff])]))
            x1_f.append(np.concatenate([np.abs(input)]))
            #if (M_abs > 1.e-15):
            #    x1_f.append(np.concatenate([np.abs(input)]))
            #else:
            #    input = np.zeros(16)
            #    x1_f.append(np.concatenate([np.abs(input)]))
            #    idx_del.append(i)


    Idx = [i for i in range(x1.shape[0])]
    idx1 = list(set(Idx) - set(idx_del)) # Delete from original data
    print("OK")
    x1_f = np.array(x1_f)
    X2 = X[idx1, :]
    x2 = x1_f[idx1,:]
    y2 = y[idx1,:]

    if (is_normed_by_stencil):
        for i in range(x1_f.shape[0]):
            M = np.max(x1_f[i, :])
            m = np.min(x1_f[i, :])
            if (M - m) > 1.e-5:
                x1_f[i, :] = (x1_f[i, :] - m)/(M - m)
            else:
                x1_f[i, :] = 0.0
                #idx_del.append(i)

    if (is_normed_by_computational_field):
        for i in range(X2.shape[0]):
            M = X2[i, N_stc+1]
            m = X2[i, N_stc+2]
            if (M - m) > 1.e-5:
                x1_f[i, :] = (x1_f[i, :] - m)/(M - m)
            else:
                x1_f[i, :] = 0.0
                #idx_del.append(i)

    print("---> Size of dataset: {}".format(len(y2)))
    counts = np.zeros(N_kernels)
    for col in range(y2.shape[1]):
        counts[col] = np.sum(y2[:, col] >= threshold_of_likelihood)
    for i in range(counts.shape[0]):
        print(f"case{i}:{int(counts[i])} ",end="")
    print("")


    print("\n2) Remove dupricate data...",end="")
    idx = list(set(i for i in range(x2.shape[0])))
    x_print = np.hstack([x2[idx, :], y2[idx, :]])

    x_print = remove_duplicates(x_print)
    x_print = np.array(x_print)
    random.shuffle(x_print)
    print("OK")

    print("---> Size of dataset: {}".format(len(x_print)))
    counts = np.zeros(N_kernels)
    for col in range(N_kernels):
        counts[col] = np.sum(x_print[:, -N_kernels+col] >= threshold_of_likelihood)
    for i in range(counts.shape[0]):
        print(f"case{i}:{int(counts[i])} ",end="")
    print("")

    # x_printをcaseごとにそれぞれN_each_cases個に制限（重複なしで抽出）
    print("\n3) Filtering...", end="")
    x_print_list = []
    used_idx = set()
    for i in range(N_kernels):
        # 各caseの条件を満たすインデックスを取得
        case_indices = [idx for idx in range(x_print.shape[0]) if x_print[idx, -N_kernels + i] >= threshold_of_likelihood and idx not in used_idx]
        # N_each_cases個だけランダムに選ぶ（足りなければ全て）
        if len(case_indices) > N_each_cases:
            selected = random.sample(case_indices, N_each_cases)
        else:
            selected = case_indices
        for idx in selected:
            x_print_list.append(x_print[idx, :])
            used_idx.add(idx)
    x_print = np.vstack(x_print_list)
    print("OK")


    print("---> Size of dataset: {}".format(len(x_print)))
    counts = np.zeros(N_kernels)
    for col in range(N_kernels):
        counts[col] = np.sum(x_print[:, -N_kernels+col] >= threshold_of_likelihood)
    for i in range(counts.shape[0]):
        print(f"case{i}:{int(counts[i])} ",end="")
    print("")
    print("")

    random.shuffle(x_print)
    all_x_print_list.append(x_print)


if all_x_print_list:
    final_x_print = np.vstack(all_x_print_list)

    print("---> Size of dataset: {}".format(len(final_x_print)))
    counts = np.zeros(N_kernels)
    for col in range(N_kernels):
        counts[col] = np.sum(final_x_print[:, -N_kernels+col] >= threshold_of_likelihood)
    for i in range(counts.shape[0]):
        print(f"case{i}:{int(counts[i])} ",end="")
    print("")

    np.savetxt(output_file, final_x_print)
    print(f"Save dataset ({output_file})")
else:
    print("No data processed.")
