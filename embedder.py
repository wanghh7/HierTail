import scipy.sparse as sp
import numpy as np
import torch
import data_load
import utils
from torch_geometric.utils.loop import add_self_loops, remove_self_loops

class embedder:
    def __init__(self, args):
        if args.gpu == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

        # Load data - Cora, CiteSeer, cora_full
        self._dataset = data_load.Dataset(root="data", dataset=args.dataset, is_normalize=args.is_normalize, add_self_loop=args.add_sl)
        self.edge_index = self._dataset.edge_index

        adj = self._dataset.adj
        features = self._dataset.features
        labels = self._dataset.labels
        class_sample_num = 20

        # Natural Setting
        if args.lt_setting == 1:
            args.criterion = 'mean'
            labels, og_to_new = utils.refine_label_order(labels)
            idx_train, idx_val, idx_test, class_num_mat = utils.split_natural(labels, og_to_new)
            samples_per_label = torch.tensor(class_num_mat[:,0])

        # Manual Setting
        elif (args.lt_setting == 2) & (args.dataset in ['amz_cloth', 'amz_eletronics']):
                args.criterion = 'mean'
                data = self._dataset
                labels = data.labels
                edge, features, labels = utils.syn_longtailed_data(data.edge_index, data.adj, data.features, labels, args.rnd, 0.8, 25, 55)

                if args.add_sl:
                    edge = remove_self_loops(edge)[0]
                    edge = add_self_loops(edge)[0]
                adj = sp.coo_matrix((np.ones(edge.shape[1]), (edge[0,:], edge[1,:])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
                adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
                self.edge_index = edge

                labels, og_to_new = utils.refine_label_order(labels)
                idx_train, idx_val, idx_test, class_num_mat = utils.split_syn_lt(labels, og_to_new, 25, 55)
                samples_per_label = torch.tensor(class_num_mat[:,0])

        self.adj = adj.to(args.device)
        self.features = features.to(args.device)
        self.labels = labels.to(args.device)
        self.class_sample_num = class_sample_num

        self.idx_train = idx_train.to(args.device)
        self.idx_val = idx_val.to(args.device)
        self.idx_test = idx_test.to(args.device)

        self.samples_per_label = samples_per_label
        self.class_num_mat = class_num_mat
        print(class_num_mat)

        args.nfeat = features.shape[1]
        args.nclass = labels.max().item() + 1

        if 'HierTail' in args.embedder:
            args.pool_ratios[0] = args.pool_ratios[0] * len(labels) / args.nclass
        else:
            im_class_num = args.im_class_num
            self.im_class_num = im_class_num

        self.args = args
