import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Leiden_GCN')

    # Dataset args
    parser.add_argument('--dataset', type=str, default='ogbn-proteins',
                        help='dataset name')
    parser.add_argument('--aggr', type=str, default='mean', choices=['mean', 'max', 'add'],
                        help='the aggregation operator to obtain nodes\' initial features')
    parser.add_argument('--clusters_path', type=str, default='dataset/leiden_clusters.csv',
                        help='the file path of graph clusters')
    parser.add_argument('--cluster_number', type=int, default=20,
                        help='number of clusters')
    parser.add_argument('--intervals', type=int, default=20,
                        help='Training epoch before clusters again')
    parser.add_argument('--cluster_type', type=str, default='random', choices=['random', 'leiden'],
                        help='Graph partition type (random or leiden)')
    parser.add_argument('--edge_sampling', type=int, default=100000000,
                        help='maximum number of edge per cluster')

    # Training args
    parser.add_argument('--nruns', type=int, default=5, help='number of training time')
    parser.add_argument('--use_cpu', action='store_true', help='Default using gpu')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train')
    parser.add_argument('--num_evals', type=int, default=1,
                        help='The number of evaluation times')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate set for optimizer')
    parser.add_argument('--dropout', type=float, default=0.5)

    # Model args
    parser.add_argument('--num_layers', type=int, default=3,
                        help='the number of layers of the networks')
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--in_channels', type=int, default=8,
                        help='the dimension of nodes/edges feature')
    parser.add_argument('--num_tasks', type=int, default=112,
                        help='the number of prediction tasks')

    # # Using node feature or not
    # parser.add_argument('--use_one_hot_encoding', action='store_true')

    # Model path to save
    parser.add_argument('--model_save_path', type=str, default='model_ckpt',
                        help='the directory used to save models')

    # Save an experiment
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')

    # Load pre-trained model
    parser.add_argument('--model_load_path', type=str, default='ogbn_proteins_pretrained_model.pth',
                        help='the path of pre-trained model')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()
    print(args)
