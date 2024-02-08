import numpy as np
import math
import torch
import random
from sklearn.metrics import f1_score, classification_report, confusion_matrix, balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
import os.path as osp
import os
import logging
import sys
from torch_scatter import scatter_mean, scatter_max
from torch_sparse import coalesce, transpose, spspmm
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, degree
import scipy.sparse as sp


def split_natural(labels, idx_map):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        idx = list(idx_map.keys())[list(idx_map.values()).index(i)]
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('OG:{:d} -> NEW:{:d}-th class sample number: {:d}'.format(idx, i, len(c_idx)))
        c_num = len(c_idx)

        if c_num == 1:
            c_num_mat[i, 0] = 1
            c_num_mat[i, 1] = 1
            c_num_mat[i, 2] = 1
            train_idx = train_idx + c_idx[:1]
            val_idx = val_idx + c_idx[:1]
            test_idx = test_idx + c_idx[:1]
        elif c_num == 2:
            c_num_mat[i, 0] = 1
            c_num_mat[i, 1] = 1
            c_num_mat[i, 2] = 1
            train_idx = train_idx + c_idx[:1]
            val_idx = val_idx + c_idx[:1]
            test_idx = test_idx + c_idx[1:2]
        elif c_num < 10:
            random.shuffle(c_idx)
            c_idxs.append(c_idx)
            c_num_mat[i,0] = math.ceil(c_num*0.1) # 10% for train
            c_num_mat[i,1] = math.ceil(c_num*0.1) # 10% for validation
            c_num_mat[i,2] = c_num - 2 * math.ceil(c_num*0.1) # 80% for test
            # print('[{}-th class] Total: {} | Train: {} | Val: {} | Test: {}'.format(i,len(c_idx), c_num_mat[i,0], c_num_mat[i,1], c_num_mat[i,2]))
            train_idx = train_idx + c_idx[:c_num_mat[i, 0]]
            val_idx = val_idx + c_idx[c_num_mat[i, 0]:c_num_mat[i, 0] + c_num_mat[i, 1]]
            test_idx = test_idx + c_idx[c_num_mat[i, 0] + c_num_mat[i, 1]:c_num_mat[i, 0] + c_num_mat[i, 1] + c_num_mat[i, 2]]
        else:
            random.shuffle(c_idx)
            c_idxs.append(c_idx)
            c_num_mat[i, 0] = int(c_num * 0.1)  # 10% for train
            c_num_mat[i, 1] = int(c_num * 0.1)  # 10% for validation
            c_num_mat[i, 2] = int(c_num * 0.8)  # 80% for test
            train_idx = train_idx + c_idx[:c_num_mat[i, 0]]
            val_idx = val_idx + c_idx[c_num_mat[i, 0]:c_num_mat[i, 0] + c_num_mat[i, 1]]
            test_idx = test_idx + c_idx[c_num_mat[i, 0] + c_num_mat[i, 1]:c_num_mat[i, 0] + c_num_mat[i, 1] + c_num_mat[i, 2]]

    random.shuffle(train_idx)
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    return train_idx, val_idx, test_idx, c_num_mat


def accuracy(output, labels, sep_point=None, sep=None, pre=None):
    if output.shape != labels.shape:
        if len(labels) == 0:
            return np.nan
        preds = output.max(1)[1].type_as(labels)
    else:
        preds= output

    correct = preds.eq(labels).double()
    correct = correct.sum()

    if correct > len(labels):
        print("wrong")
    return correct / len(labels)

def classification(output, labels, sep_point=None, sep=None):
    target_names = []
    if len(labels) == 0:
        return np.nan
    else:
        pred = output.max(1)[1].type_as(labels)
        for i in labels.unique():
            target_names.append(f'class_{int(i)}')
        return classification_report(labels, pred)

def confusion(output, labels, sep_point=None, sep=None):
    if len(labels) == 0:
        return np.nan
    else:
        
        pred = output.max(1)[1].type_as(labels)
        return confusion_matrix(labels, pred)

def performance_measure(output, labels, sep_point=None, sep=None, pre=None):
    acc = accuracy(output, labels, sep_point=sep_point, sep=sep, pre=pre)*100

    if len(labels) == 0:
        return np.nan
    
    if output.shape != labels.shape:
        output = torch.argmax(output, dim=-1)

    macro_F = f1_score(labels.cpu().detach(), output.cpu().detach(), average='macro')*100
    gmean = geometric_mean_score(labels.cpu().detach(), output.cpu().detach(), average='macro')*100
    bacc = balanced_accuracy_score(labels.cpu().detach(), output.cpu().detach())*100
    return acc, macro_F, gmean, bacc

def adj_mse_loss(adj_rec, adj_tgt, adj_mask = None):
    adj_tgt[adj_tgt != 0] = 1

    edge_num = adj_tgt.nonzero().shape[0] #number of non-zero
    total_num = adj_tgt.shape[0]**2 #possible edge

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2) # element-wise
    return loss

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)

def refine_label_order(labels):
    print('Refine label order, Many to Few')
    num_labels = labels.max() + 1
    num_labels_each_class = np.array([(labels == i).sum().item() for i in range(num_labels)])
    sorted_index = np.argsort(num_labels_each_class)[::-1]
    idx_map = {sorted_index[i]:i for i in range(num_labels)}
    new_labels = np.vectorize(idx_map.get)(labels.numpy())
    return labels.new(new_labels), idx_map

def normalize_output(out_feat, idx):
    sum_m = 0
    for m in out_feat:
        sum_m += torch.mean(torch.norm(m[idx], dim=1))
    return sum_m 

def normalize_adj(adj):
    """Row-normalize sparse matrix"""
    deg = torch.sum(adj.to_dense(), dim=1)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    deg_inv_sqrt = torch.diag(deg_inv_sqrt).to_sparse()
    adj = torch.spmm(deg_inv_sqrt, adj.to_dense()).to_sparse()
    return adj

def normalize_sym(adj):
    """Symmetric-normalize sparse matrix"""
    deg = torch.sum(adj.to_dense(), dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    deg_inv_sqrt = torch.diag(deg_inv_sqrt).to_sparse()

    adj = torch.spmm(deg_inv_sqrt, adj.to_dense()).to_sparse()
    adj = torch.spmm(adj, deg_inv_sqrt.to_dense()).to_sparse()
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch_sparse_tensor_to_sparse_mx(torch_sparse, shape):
    """Convert a torch sparse tensor to a scipy sparse matrix."""
    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()
    sp_matrix = sp.coo_matrix((data, (row, col)), shape=shape, dtype=np.float32)
    return sp_matrix

def scheduler(epoch, curriculum_ep=500, func='convex'):
    if func == 'convex':
        return np.cos((epoch * np.pi) / (curriculum_ep * 2))
    elif func == 'concave':
        return np.power(0.99, epoch)
    elif func == 'linear':
        return 1 - (epoch / curriculum_ep)
    elif func == 'composite':
        return (1/2) * np.cos((epoch*np.pi) / curriculum_ep) + 1/2

def setupt_logger(save_dir, text, filename = 'log.txt'):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(text)
    # for each in logger.handlers:
    #     logger.removeHandler(each)
    logger.setLevel(4)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info("======================================================================================")
    return logger


def balance_contrastive(embed, centers, labels, args):
    if centers != '':
        embed = torch.cat([embed, centers], dim=0)
        labels = torch.cat([labels, torch.arange(end=centers.shape[0], device=args.device)], dim=0)
    dot_product_all = torch.div(torch.mm(embed, embed.T), args.temperature)
    exp_dot_all = (torch.exp(dot_product_all - torch.max(dot_product_all, dim=1, keepdim=True)[0]) + 1e-7)
    mask_same_class = labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels

    same_cls_count = torch.sum(mask_same_class, 1)
    denominator = torch.sum(torch.div(exp_dot_all, same_cls_count.T), 1).unsqueeze(1).repeat(1, exp_dot_all.shape[0])
    frac = torch.log(torch.div(exp_dot_all, denominator)) * mask_same_class

    loss_bcl = - torch.div(torch.sum(frac, 1), same_cls_count)
    return loss_bcl.mean()


def StAS(index_A, value_A, index_S, value_S, device, N, kN):
    r"""StAS: a function which returns new edge weights for the pooled graph using the formula S^{T}AS"""

    index_A, value_A = coalesce(index_A, value_A, m=N, n=N)
    index_S, value_S = coalesce(index_S, value_S, m=N, n=kN)
    index_B, value_B = spspmm(index_A, value_A, index_S, value_S, N, N, kN)

    index_St, value_St = transpose(index_S, value_S, N, kN)
    index_B, value_B = coalesce(index_B, value_B, m=N, n=kN)
    # index_E, value_E = spspmm(index_St.cpu(), value_St.cpu(), index_B.cpu(), value_B.cpu(), kN, N, kN)
    index_E, value_E = spspmm(index_St, value_St, index_B, value_B, kN, N, kN)

    # return index_E.to(device), value_E.to(device)
    return index_E, value_E


def graph_connectivity(device, perm, edge_index, edge_weight, score, ratio, batch, N):
    r"""graph_connectivity: is a function which internally calls StAS func to maintain graph connectivity"""

    kN = perm.size(0)
    perm2 = perm.view(-1, 1)

    # mask contains bool mask of edges which originate from perm (selected) nodes
    mask = (edge_index[0] == perm2).sum(0, dtype=torch.bool)

    # create the S
    S0 = edge_index[1][mask].view(1, -1)
    S1 = edge_index[0][mask].view(1, -1)
    index_S = torch.cat([S0, S1], dim=0)
    value_S = score[mask].detach().squeeze()

    # relabel for pooling ie: make S [N x kN]
    n_idx = torch.zeros(N, dtype=torch.long)
    n_idx[perm] = torch.arange(perm.size(0))
    index_S[1] = n_idx[index_S[1]]

    # create A
    index_A = edge_index.clone()
    if edge_weight is None:
        value_A = value_S.new_ones(edge_index[0].size(0))
    else:
        value_A = edge_weight.clone()

    fill_value = 1
    index_E, value_E = StAS(index_A, value_A, index_S, value_S, device, N, kN)
    index_E, value_E = remove_self_loops(edge_index=index_E, edge_attr=value_E)
    index_E, value_E = add_remaining_self_loops(edge_index=index_E, edge_attr=value_E,
                                                fill_value=fill_value, num_nodes=kN)
    return index_E, value_E


def readout(x, batch):
    x_mean = scatter_mean(x, batch, dim=0)
    x_max, _ = scatter_max(x, batch, dim=0)
    return torch.cat((x_mean, x_max), dim=-1)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_sp_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def syn_longtailed_data(edge_index, adj, features, labels, seed, alpha, n_val, n_test):
    # make dataset following long-tail distribution
    num_classes = len(set(labels.tolist()))
    n_data = []
    for i in range(num_classes):
        data_num = (labels == i).sum()
        n_data.append(int(data_num.item()))
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    _, tmp = torch.sort(indices)
    sorted_n_train = sorted_n_data - n_val - n_test
    n_head = sum(sorted_n_train[:int((1 - alpha) * num_classes)])
    n_tail = sum(sorted_n_train[int((1 - alpha) * num_classes):])
    sorted_n_train[int((1 - alpha) * num_classes):] = sorted_n_train[int((1 - alpha) * num_classes):] * n_head / n_tail * (1-alpha) / alpha
    sorted_n_data = sorted_n_train + n_val + n_test
    n_data = sorted_n_data[tmp]

    node_d = degree(edge_index[0], num_nodes=len(labels))

    idx_list = []
    for i in range(num_classes):
        c_idx = (labels == i).nonzero()[:, -1].tolist()
        tmp = node_d[c_idx]
        sort_zipped = sorted(zip(c_idx, tmp), key=lambda x: (x[1], x[0]), reverse=True)
        c_idx, _ = [list(x) for x in zip(*sort_zipped)]
        c_idx = c_idx[:n_data[i]]
        random.seed(seed)
        random.shuffle(c_idx)
        idx_list = idx_list + c_idx
    assert len(set(idx_list)) == len(idx_list)

    # removed nodes and edges
    node_mask = labels.new_zeros(labels.size(), dtype=torch.bool)
    node_mask[idx_list] = True

    adj = torch.index_select(adj, 0, torch.tensor(idx_list))
    adj = torch.index_select(adj, 1, torch.tensor(idx_list))

    edge = (adj.to_dense() > 0).nonzero().t()
    node = features[node_mask]
    label = labels[node_mask]
    return edge, node, label


def split_syn_lt(labels, idx_map, n_val, n_test):
    num_classes = len(set(labels.tolist()))
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    c_num_mat[:,1] = n_val
    c_num_mat[:,2] = n_test

    for i in range(num_classes):
        idx = list(idx_map.keys())[list(idx_map.values()).index(i)]
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('OG:{:d} -> NEW:{:d}-th class sample number: {:d}'.format(idx, i, len(c_idx)))
        c_num_mat[i,0] = len(c_idx) - n_val - n_test

        train_idx = train_idx + c_idx[:c_num_mat[i, 0]]
        val_idx = val_idx + c_idx[c_num_mat[i, 0]:c_num_mat[i, 0] + c_num_mat[i, 1]]
        test_idx = test_idx + c_idx[c_num_mat[i, 0] + c_num_mat[i, 1]:c_num_mat[i, 0] + c_num_mat[i, 1] + c_num_mat[i, 2]]

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, c_num_mat

