from embedder import embedder
import torch.nn as nn
import torch.optim as optim
import utils
import torch.nn.functional as F
import layers
import numpy as np
from copy import deepcopy
import os
import time
from sklearn.metrics import pairwise_distances_argmin


class HierTail():
    def __init__(self, args):
        self.args = args

    def training(self):
        self.args.embedder = self.args.embedder + f'_{self.args.ndpth}-depths' + f'_ratio-{self.args.pool_ratios}' + \
                             f'_nhid-{self.args.nhid}' + f'_wght-{self.args.weight_cpc}' + f'_tmpt-{self.args.temperature}'
        if self.args.lt_setting == 1:  # natural
            os.makedirs(f'./results/baseline/natural/{self.args.dataset}', exist_ok=True)
            text = open(f'./results/baseline/natural/{self.args.dataset}/{self.args.embedder}.txt', 'w')
        else:  # manual
            os.makedirs(f'./results/baseline/manual/{self.args.dataset}/{self.args.lt_setting}',
                        exist_ok=True)
            text = open(
                f'./results/baseline/manual/{self.args.dataset}/{self.args.lt_setting}/{self.args.embedder}.txt', 'w')

        seed_result = {}
        seed_result['acc'] = []
        seed_result['macro_F'] = []
        seed_result['gmeans'] = []
        seed_result['bacc'] = []

        for seed in range(self.args.rnd, self.args.rnd + self.args.num_seed):
            print(f'============== seed:{seed} ==============')
            t_total = time.time()
            utils.seed_everything(seed)
            print('seed:', seed, file=text)
            self = embedder(self.args)

            model = modeler(self.args).to(self.args.device)
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr,
                                      weight_decay=self.args.wd)

            # Main training
            val_f = []
            test_results = []

            best_metric = 0

            for epoch in range(self.args.ep):
                model.train()
                optimizer.zero_grad()

                loss = model(self.features, self.edge_index, self.labels, self.idx_train)

                loss.backward()

                optimizer.step()

                # Evaluation
                model.eval()
                # with torch.no_grad():
                output, _, _ = model.unet(self.features, self.edge_index.to(self.args.device))

                acc_val, macro_F_val, gmeans_val, bacc_val = utils.performance_measure(output[self.idx_val],
                                                                                       self.labels[self.idx_val],
                                                                                       pre='valid')

                val_f.append(macro_F_val)
                max_idx = val_f.index(max(val_f))

                if best_metric <= macro_F_val:
                    best_metric = macro_F_val
                    best_model = deepcopy(model)

                # Test
                acc_test, macro_F_test, gmeans_test, bacc_test = utils.performance_measure(output[self.idx_test],
                                                                                           self.labels[self.idx_test],
                                                                                           pre='test')

                test_results.append([acc_test, macro_F_test, gmeans_test, bacc_test])
                best_test_result = test_results[max_idx]

                st = "[seed {}][{}][Epoch {}]".format(seed, self.args.embedder, epoch)
                st += "[Val] ACC: {:.3f}, Macro-F1: {:.3f}, G-Means: {:.3f}, bACC: {:.3f}|| ".format(acc_val,
                                                                                                     macro_F_val,
                                                                                                     gmeans_val,
                                                                                                     bacc_val)
                st += "[Test] ACC: {:.3f}, Macro-F1: {:.3f}, G-Means: {:.3f}, bACC: {:.3f}\n".format(acc_test,
                                                                                                     macro_F_test,
                                                                                                     gmeans_test,
                                                                                                     bacc_test)
                st += "  [*Best Test Result*][Epoch {}] ACC: {:.3f}, Macro-F1: {:.3f}, G-Means: {:.3f}, bACC: {:.3f}".format(
                    max_idx, best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3])

                if epoch % 100 == 0:
                    print(st, file=text)
                    print(st)

                if (epoch - max_idx > self.args.ep_early) or (epoch + 1 == self.args.ep):
                    if epoch - max_idx > self.args.ep_early:
                        print("Early stop", file=text)

                    # with torch.no_grad():
                    output, _, _ = best_model.unet(self.features, self.edge_index.to(self.args.device))

                    best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[
                        3] = utils.performance_measure(output[self.idx_test], self.labels[self.idx_test], pre='test')
                    acc_list, macro_F_list, gmean_list, bacc_list = utils.performance_per_class(output[self.idx_test], self.labels[self.idx_test], pre='test')
                    print("[Best Test Result] ACC: {:.3f}, Macro-F1: {:.3f}, G-Means: {:.3f}, bACC: {:.3f}".format(
                        best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3]), file=text)
                    print(utils.classification(output[self.idx_test], self.labels[self.idx_test].detach().cpu()),
                          file=text)
                    print(utils.confusion(output[self.idx_test], self.labels[self.idx_test].detach().cpu()), file=text)
                    print(file=text)
                    break

            seed_result['acc'].append(float(best_test_result[0]))
            seed_result['macro_F'].append(float(best_test_result[1]))
            seed_result['gmeans'].append(float(best_test_result[2]))
            seed_result['bacc'].append(float(best_test_result[3]))

        acc = seed_result['acc']
        f1 = seed_result['macro_F']
        gm = seed_result['gmeans']
        bacc = seed_result['bacc']

        print(
            '[Averaged result] ACC: {:.3f}+{:.3f}, Macro-F: {:.3f}+{:.3f}, G-Means: {:.3f}+{:.3f}, bACC: {:.3f}+{:.3f}'.format(
                np.mean(acc), np.std(acc), np.mean(f1), np.std(f1), np.mean(gm), np.std(gm), np.mean(bacc),
                np.std(bacc)))
        print(file=text)
        print('ACC Macro-F G-Means bACC', file=text)
        print('{:.3f}+{:.3f} {:.3f}+{:.3f} {:.3f}+{:.3f} {:.3f}+{:.3f}'.format(np.mean(acc), np.std(acc), np.mean(f1),
                                                                               np.std(f1), np.mean(gm), np.std(gm),
                                                                               np.mean(bacc), np.std(bacc)), file=text)
        print(file=text)
        print(self.args, file=text)
        print(self.args)
        text.close()


class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        if self.args.activation == 'tanh':
            act = F.tanh
        elif self.args.activation == 'sigmoid':
            act = F.sigmoid
        else:
            act = F.relu

        self.unet = layers.GraphUNet(in_channels=args.nfeat, hidden_channels=args.nhid, out_channels=args.nclass, depth=args.ndpth,
                                     gnn_type=args.gnn_type, pool_ratios=args.pool_ratios, act=act, dropout=args.dropout)

    def forward(self, feature, edge_index, labels, idx_train):
        if edge_index.device != feature.device:
            edge_index = edge_index.to(feature.device)

        output, xs, perms = self.unet(feature, edge_index)
        loss_nodeclassification = F.cross_entropy(output[idx_train], labels[idx_train])

        loss_scl = utils.balance_contrastive(output[idx_train], '', labels[idx_train], self.args)

        loss_bcl = 0
        for i in range(self.args.ndpth - 1):
            X = xs[0]
            Y = xs[i + 1]
            idx = pairwise_distances_argmin(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), axis=1, metric='euclidean')
            cluster_labels = perms[i][idx]
            loss_bcl += utils.balance_contrastive(X, Y, cluster_labels, self.args)

        return loss_nodeclassification + (loss_scl + loss_bcl) * self.args.weight_cpc

