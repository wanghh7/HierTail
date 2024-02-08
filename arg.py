import argparse

def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedder', nargs='?', default='origin', choices=['HierTail', 'origin', 'reweight', 'oversampling',
                                                                            'smote', 'embed_smote', 'graphsmote_T', 'graphsmote_O'])
    parser.add_argument('--gpu', type=int, default=0, help="Choose GPU number")
    parser.add_argument('--dataset', type=str, default='cora_full', choices=['cora_full', 'email', 'wiki', 'blog', 'amz_cloth', 'amz_eletronics'])
    parser.add_argument('--lt_setting', type=float, default=1, help="1 for natural, 2 for manual")
    parser.add_argument('--layer', type=str, default='gcn', choices=['gcn', 'gat'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--ep_pre', type=int, default=0, help="Number of epochs to pretrain.")
    parser.add_argument('--ep', type=int, default=10000, help="Number of epochs to train.")
    parser.add_argument('--ep_early', type=int, default=1000, help="Early stop criterion.")
    parser.add_argument('--add_sl', type=str2bool, default=True, help="Whether to include self-loop")
    parser.add_argument('--adj_norm_1', action='store_true', default=True, help="D^(-1)A")
    parser.add_argument('--adj_norm_2', action='store_true', default=False, help="D^(-1/2)AD^(-1/2)")
    parser.add_argument('--nhead', type=int, default=1, help="Number of multi-heads")
    parser.add_argument('--nhid', type=int, default=64, help="Number of hidden dimensions")
    parser.add_argument('--wd', type=float, default=5e-4, help="Controls weight decay")
    parser.add_argument('--num_seed', type=int, default=10, help="Number of total seeds")
    parser.add_argument('--rnd', type=int, default=123, help="Random seed")
    parser.add_argument('--is_normalize', action='store_true', default=False, help="Normalize features")
    parser.add_argument('--cls_og', type=str, default='GNN', choices=['GNN', 'MLP'], help="Wheter to user (GNN+MLP) or (MLP) as a classifier")

    if parser.parse_known_args()[0].embedder == 'HierTail':
        parser.add_argument('--ndpth', type=int, default=3, help='The depth of the task grouping.')
        parser.add_argument('--pool_ratios', nargs='+', type=float, default='0.5', help="Task grouping ratio for each depth.")
        parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'sigmoid'], help="Activation Function")
        parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gcn2'], help="GCN type according to the order of GCN")
        parser.add_argument('--weight_cpc', type=float, default=1.0, help="Weight of node classification and contrastive learning")
        parser.add_argument('--temperature', type=float, default=0.1, help="Scalar temperature controls tolerance to similar samples")
    else:
        parser.add_argument('--up_scale', type=float, default=1, help="Scale of oversampling")
        parser.add_argument('--im_class_num', type=int, default=3, help="Number of tail classes")
        parser.add_argument('--rw', type=float, default=0.000001, help="Balances edge loss within node classification loss")

    return parser