import copy
import time
import logging
import glob
import torch
import torch.optim as optim
from dataset import OGBNDataset
from ogb.nodeproppred import Evaluator
from utils.args import args_parser
from utils.logger import Logger
from utils.cluster import (
    cluster_reader,
    random_partition_graph,
    build_graph,
    leiden_clustering,
    custom_clustering
)
from utils.loops import (
    train,
    multi_evaluate)
from gcn import GCN
import warnings

warnings.filterwarnings('ignore')


def training(args, scripts=True):
    if scripts:
        logger = Logger(args, scripts_to_save=glob.glob('*.py'))
    else:
        logger = Logger(args)
    args = logger.args
    logging.info(f"Arguments: {args}")

    if not args.use_cpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    logging.info(f"Training device: {device}")

    dataset = OGBNDataset(dataset_name=args.dataset, aggr=args.aggr)

    evaluator = Evaluator(args.dataset)
    criterion = torch.nn.BCEWithLogitsLoss()

    sub_dir = 'random-train_{}-test_{}-num_evals_{}'.format(args.cluster_number,
                                                            args.cluster_number,
                                                            args.num_evals)
    logging.info(f"Model filename used to save: {sub_dir}")

    # GCN
    model = GCN(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=10,
                                                           min_lr=1e-6, verbose=True)

    results = {'highest_valid': (0, 0),
               'final_train': (0, 0),
               'final_test': (0, 0),
               'highest_train': (0, 0)}

    start_time = time.time()

    if args.cluster_type == 'random':
        parts = random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.cluster_number)
    else:
        parts = cluster_reader(args.cluster_path)
    args.cluster_number = len(parts)
    data = dataset.generate_sub_graphs(parts, cluster_number=args.cluster_number)

    for epoch in range(1, args.epochs + 1):

        if epoch % args.intervals == 0:
            print("\n==========Sampling subgraph for next interval==========")
            if args.cluster_type == 'random':
                parts = random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.cluster_number)
            else:
                G = build_graph('./dataset')
                leiden_parts = leiden_clustering(G, None)
                parts = custom_clustering(G, leiden_parts, threshold=10000, n_iters=1)
            args.cluster_number = len(parts)
            data = dataset.generate_sub_graphs(parts, cluster_number=args.cluster_number)

        epoch_loss = train(data, dataset, model, optimizer, criterion, device, edge_sampling=args.edge_sampling)
        # logging.info('Epoch {}, training loss {:.4f}'.format(epoch, epoch_loss))

        result = multi_evaluate([data], dataset, model, evaluator, device, edge_sampling=args.edge_sampling * 2)

        scheduler.step(result['valid']['rocauc'])

        logger.add_results(result, epoch_loss)
        logger.print_statistic()

        train_result = result['train']['rocauc']
        valid_result = result['valid']['rocauc']
        test_result = result['test']['rocauc']

        if valid_result > results['highest_valid'][0]:
            results['highest_valid'] = (valid_result, epoch)
            results['final_train'] = (train_result, epoch)
            results['final_test'] = (test_result, epoch)

            logger.save_ckpt(model, optimizer, round(epoch_loss, 4), epoch, sub_dir, name_post='valid_best')

        if train_result > results['highest_train'][0]:
            results['highest_train'] = (train_result, epoch)

    logging.info(f"Final results: {results}")

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))


def main():
    args = args_parser()
    for run in range(args.nruns):
        print(f"\n\nTimes of experiments: {run+1}\n\n")
        cur_args = copy.deepcopy(args)
        if run:
            training(cur_args, scripts=False)
        else:
            training(cur_args)


if __name__ == '__main__':
    main()
