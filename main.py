import copy
import os
import time
import logging
import glob
import torch
import torch.optim as optim
from dataset import OGBNDataset
from ogb.nodeproppred import Evaluator
from utils.args import args_parser
from utils.logger import Logger, set_logging
from utils.cluster import partition_graph, node_species_histogram, histogram_plot
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

    sub_dir = '{}-train_{}-test_{}-num_evals_{}'.format(args.cluster_type, args.cluster_number,
                                                        args.cluster_number,
                                                        args.num_evals)
    logging.info(f"Model filename used to save: {sub_dir}")

    # GCN
    model = GCN(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=250,
                                                                     eta_min=1e-6,
                                                                     verbose=True)

    results = {'highest_valid': (0, 0),
               'final_train': (0, 0),
               'final_test': (0, 0),
               'highest_train': (0, 0)}

    start_time = time.time()
    # args.clusters_path = os.path.splitext(args.clusters_path)[0]

    data_train = None
    for epoch in range(1, args.epochs + 1):

        if epoch % args.intervals == 1 or args.intervals == 1:
            logging.info("\n==========Sampling subgraph for next interval==========")
            mini_batches_train = partition_graph(dataset, args)
            data_train = dataset.generate_sub_graphs(mini_batches_train, cluster_number=args.cluster_number)

            if epoch == 1:
                # Visualize training clusters
                mini_batches_histogram_train = node_species_histogram(dataset.whole_graph, mini_batches_train)
                histogram_plot(mini_batches_histogram_train, dataset.whole_graph.node_species.unique(),
                               label="train_histogram")

            args.cluster_number = len(mini_batches_train)
            logging.info(f"Number of nodes in batches: {sum(len(c) for c in mini_batches_train)} - "
                         f"Number of nodes in graph: {len(set(c for C in mini_batches_train for c in C))}\n")

        epoch_loss = train(data_train, dataset, model, optimizer, criterion, device)

        result = multi_evaluate([data_train], dataset, model, evaluator, device)

        scheduler.step()

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
        print(f"\n\nTimes of experiments: {run + 1}\n\n")
        cur_args = copy.deepcopy(args)
        if run:
            training(cur_args, scripts=False)
        else:
            training(cur_args)


if __name__ == '__main__':
    main()
