import csv
import os
import numpy as np
import leidenalg as la
import time
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm
import igraph as ig
from copy import deepcopy
from bisect import insort_right
import torch
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import math


def generate_mini_batch(parts, num_cluster, overlap=0, dataset=None):
    part_per_batch = len(parts) // num_cluster

    random_idx = np.arange(len(parts))
    np.random.shuffle(random_idx)
    mini_batches = []
    for i in range(num_cluster - 1):
        mini_batch = []
        for parts_idx in range(i * part_per_batch, (i + 1) * part_per_batch):
            mini_batch.extend(parts[random_idx[parts_idx]])

        if overlap:
            mini_batch = overlapping_batch(dataset, mini_batch, overlap=overlap)
        mini_batches.append(mini_batch)

    last_batch = []
    for parts_idx in range((num_cluster - 1) * part_per_batch, len(parts)):
        last_batch.extend(parts[random_idx[parts_idx]])

    if overlap:
        last_batch = overlapping_batch(dataset, last_batch, overlap=overlap)
    mini_batches.append(last_batch)

    return mini_batches


def overlapping_batch(dataset, cluster, overlap=300):
    data = dataset[0]
    train_node = dataset.get_idx_split()['train']
    train_species = data.node_species[dataset.get_idx_split()['train']]

    batch_node = torch.tensor(intersection(dataset.get_idx_split()['train'].tolist(), cluster))
    batch_species = data.node_species[batch_node]
    count = Counter(batch_species.squeeze().tolist())

    overlap_cluster = deepcopy(cluster)
    for species in train_species.unique().tolist():
        if count.get(species, 0) >= overlap:
            continue

        train_node_same_species = train_node[torch.where(train_species == species)[0]].squeeze()
        overlap_node = np.random.choice(train_node_same_species, overlap, replace=False)
        overlap_cluster.extend(overlap_node.tolist())

    return sorted(set(overlap_cluster))


def generate_mini_batch_separately(dataset, train_cluster, valid_cluster, test_cluster, num_cluster, overlap=300):
    part_per_batch_train = len(train_cluster) // num_cluster
    part_per_batch_valid = len(valid_cluster) // num_cluster
    part_per_batch_test = len(test_cluster) // num_cluster

    random_idx_train = np.arange(len(train_cluster))
    random_idx_valid = np.arange(len(valid_cluster))
    random_idx_test = np.arange(len(test_cluster))
    np.random.shuffle(random_idx_train)
    np.random.shuffle(random_idx_valid)
    np.random.shuffle(random_idx_test)

    mini_batches = []
    for i in range(num_cluster - 1):
        mini_batch = []
        for parts_idx in range(i * part_per_batch_train, (i + 1) * part_per_batch_train):
            mini_batch.extend(train_cluster[random_idx_train[parts_idx]])
        for parts_idx in range(i * part_per_batch_valid, (i + 1) * part_per_batch_valid):
            mini_batch.extend(valid_cluster[random_idx_valid[parts_idx]])
        for parts_idx in range(i * part_per_batch_test, (i + 1) * part_per_batch_test):
            mini_batch.extend(test_cluster[random_idx_test[parts_idx]])

        mini_batch = overlapping_batch(dataset, mini_batch, overlap=overlap)
        mini_batches.append(mini_batch)

    last_batch = []
    for parts_idx in range((num_cluster - 1) * part_per_batch_train, len(train_cluster)):
        last_batch.extend(train_cluster[random_idx_train[parts_idx]])
    for parts_idx in range((num_cluster - 1) * part_per_batch_valid, len(valid_cluster)):
        last_batch.extend(valid_cluster[random_idx_valid[parts_idx]])
    for parts_idx in range((num_cluster - 1) * part_per_batch_test, len(test_cluster)):
        last_batch.extend(test_cluster[random_idx_test[parts_idx]])

    last_batch = overlapping_batch(dataset, last_batch, overlap=overlap)
    mini_batches.append(last_batch)

    return mini_batches


def leiden_partition_graph(dataset, args):
    leiden_parts = cluster_reader(f"{args.clusters_path}")

    if not args.overlap:
        mini_batches = generate_mini_batch(leiden_parts, args.cluster_number)
    else:
        mini_batches = generate_mini_batch(leiden_parts, args.cluster_number, overlap=args.overlap, dataset=dataset)

    return mini_batches


def leiden_partition_graph_by_species(dataset, args, overlap_ratio=0.1):
    all_coms_leiden = cluster_reader(f"{args.clusters_path}")
    data = dataset.whole_graph

    for idx in range(len(all_coms_leiden)):
        all_coms_leiden[idx] = np.array(all_coms_leiden[idx])

    coms_by_species = [[] for _ in range(data.node_species.unique().shape[0])]
    mapping_species = {species.item(): idx for idx, species in enumerate(data.node_species.unique())}
    node_species = data.node_species
    for com in all_coms_leiden:
        com_species = node_species[com].squeeze().tolist()
        max_cnt_com = Counter(com_species).most_common(3)
        for com_most_species in max_cnt_com:
            if overlap_ratio < 1e-8:
                flag = com_most_species[0] == max_cnt_com[0][0]
            else:
                flag = com_most_species[1] / max_cnt_com[0][1] >= (1-overlap_ratio)
            if flag:
                com_to_idx = mapping_species[com_most_species[0]]
                coms_by_species[com_to_idx].append(com)

    cluster_by_species = [set() for _ in range(args.cluster_number)]
    for species in node_species.unique():
        species_coms = deepcopy(coms_by_species[mapping_species[species.item()]])

        coms_per_batch = len(species_coms) // len(cluster_by_species)

        species_coms_idx = np.arange(len(species_coms))
        np.random.shuffle(species_coms_idx)

        for idx in range(len(cluster_by_species)):
            batch_idx = species_coms_idx[idx * coms_per_batch:(idx + 1) * coms_per_batch]
            coms_in_batch = [species_coms[i].tolist() for i in range(len(species_coms)) if i in batch_idx]
            cluster_by_species[idx].update(itertools.chain.from_iterable(coms_in_batch))

        for com_idx in range(coms_per_batch*len(cluster_by_species), len(species_coms)):
            clus_idx = np.random.randint(len(cluster_by_species), size=1)[0]
            cluster_by_species[clus_idx].update(species_coms[species_coms_idx[com_idx]])

    cluster_by_species = [sorted(list(c)) for c in cluster_by_species]

    return cluster_by_species


def partition_graph(dataset, args, infer=False):
    params = deepcopy(args)
    if infer:
        params.cluster_number = params.cluster_number//2

    if params.cluster_type == 'random':
        mini_batches = random_partition_graph(dataset.total_no_of_nodes, cluster_number=params.cluster_number)
    elif params.cluster_type == 'leiden':
        mini_batches = leiden_partition_graph(dataset.dataset, params)
    else:
        mini_batches = leiden_partition_graph_by_species(dataset, params, overlap_ratio=args.overlap_ratio)

    return mini_batches


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def difference(lst1, lst2):
    return list(set(lst1) - set(lst2))


def process_indexes(idx_list):
    """
    Find indexes of 'idx_list' list sorted by ascending value
    :param idx_list: list of value want to sort
    :return: index list sort by ascending value
    """
    idx_dict = {}
    for i, idx in enumerate(idx_list):
        idx_dict[idx] = i

    return [idx_dict[i] for i in sorted(idx_dict.keys())]


def cluster_reader(filename):
    """
    Reading clusters save in csv file. Each row in csv file represent a cluster
    :param filename: Name of the file store all clusters
    :return: a nested list contain all clusters
    """
    clusters = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            clusters.append(list(map(int, line)))
    return clusters


def cluster_writer(clusters, filename):
    """
    Writing clusters in the form of a nested list to a csv file
    :param clusters: a nested list represent all clusters
    :param filename: name of the file used to save clusters
    :return: None
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for cluster in clusters:
            writer.writerow(cluster)


def build_graph(root='./'):
    """
    Upload ogbn-proteins dataset in root directory and
    transform into igraph object for the convenience of using the leiden algorithm
    :param root: directory contain ogbn-proteins dataset
    :return: igraph object represent ogbn-proteins dataset
    """
    ds = PygNodePropPredDataset('ogbn-proteins', root=root)
    edges = ds.data.edge_index.T.numpy()
    attr = ds.data.edge_attr.numpy()

    list_edges = []
    start = time.time()
    for u, v in tqdm(edges, desc='Building edge'):
        list_edges.append((u, v))

    graph = ig.Graph(ds.data.y.shape[0], list_edges, edge_attrs={'weight': np.max(attr, axis=1)})
    graph.vs["label"] = ds.data.y

    for i in tqdm(range(graph.vcount()), desc="Building vertices"):
        graph.vs[i]['name'] = i

    print(f"Build ogbn-proteins igraph in {time.time() - start:.4f} seconds")

    return graph


def random_partition_graph(num_nodes, cluster_number=100):
    """
    Random partition all nodes into 'cluster_number' clusters
    :param num_nodes: number of nodes in graph
    :param cluster_number: number of subgraph after partition
    :return: an index array of size (num_nodes,). Values from [0, cluster_number)
    """
    parts = np.random.randint(cluster_number, size=num_nodes)
    list_parts = []
    for cluster in range(cluster_number):
        list_parts.append(np.where(parts == cluster)[0])
    return list_parts


def leiden_clustering(graph, filename=None, verbose=False, max_comm_size=0):
    """
    Apply leiden clustering algorithm for a graph
    :param graph: graph that need to cluster
    :param filename: name of the file to save clusters after using leiden algorithms
    :param verbose: if true => tracing progress by printing output
    :param max_comm_size: maximum number of nodes in each cluster
    :return: List of clusters. Each element is a list of nodes in same cluster
    """
    start = time.time()
    parts = la.find_partition(graph, la.ModularityVertexPartition, max_comm_size=max_comm_size, weights='weight')
    if len(parts) == 1:
        print(f"\033[93mWarning: Subgraph contain only 1 cluster\033[0m")
    if verbose:
        print(f"Clustering {graph.vcount()} nodes into {len(parts)} clusters in {(time.time() - start):.4f} seconds")
    if filename is not None:
        cluster_writer(parts, filename)
    return sorted(list(parts), key=lambda x: len(x))


def leiden_recursive(graph, orig_parts, max_comm_size=10000, verbose=False):
    """
    Recursively apply leiden algorithm for clusters that has larger number of nodes than a threshold
    :param graph: original graph
    :param orig_parts: list of clusters after using some clustering algorithms.
                        Each element is a list of nodes in same cluster
    :param max_comm_size: maximum number of nodes in a same cluster
    :param verbose: if true => tracing progress by printing output
    :return: list of clusters. Each element is a list of nodes in same cluster (with size smaller than threshold)
    """
    parts = deepcopy(orig_parts)
    idx = 0
    while idx < len(parts):
        if len(parts[idx]) > max_comm_size:
            subG = graph.subgraph(parts[idx])
            if verbose:
                print("Start clustering")
            sub_parts = leiden_clustering(subG, verbose=verbose)
            if len(sub_parts) == 1:
                print(f"\033[93mWarning: Divided again\033[0m")
                sub_parts = leiden_clustering(subG, verbose=True, max_comm_size=max_comm_size)
            sub_parts = [[subG.vs[p]['name'] for p in P] for P in sub_parts]
            del parts[idx]
            parts.extend(sub_parts)
        else:
            idx += 1

    parts = sorted(parts, key=lambda x: len(x))
    return parts


def outer_edge_ratio(graph, first_part, second_part, eps=1e-8):
    """
    Compute the ratio between number of edges that connect 2 clusters and total number of nodes in 2 clusters
    :param graph: original graph
    :param first_part: list of vertex in first cluster
    :param second_part: list of vertex in second cluster
    :param eps: Avoid 0 ratio
    :return: a ratio represents the degree gain if merge 2 cluster together
    """
    first = graph.subgraph(first_part)
    second = graph.subgraph(second_part)
    union = graph.subgraph(first_part + second_part)

    inner = first.ecount() + second.ecount()
    outer = union.ecount()
    return (outer - inner) / (len(first_part) + len(second_part)) + eps


def singleton_communities(parts, size=3):
    """
    Finding all community that is considered to be a singleton community
    :param parts: list of communities
    :param size: minimum size of communities that is considered not to be a singleton community
    :return: list of singleton community
    """
    single_list = []
    for idx in range(len(parts)):
        if len(parts[idx]) < size:
            single_list.extend(parts[idx])
    return single_list


def merge_clusters(graph, orig_parts, n_iters=2, max_comm_size=10000, min_comm_size=7500, verbose=False):
    """
    Merging clusters that are too small based on number of edges between clusters.
    A cluster is merging with a different cluster whose has the highest ratio between number of edges connecting
     those 2 cluster and total number of nodes of clusters
    :param graph: original graph
    :param orig_parts: list contain all clusters in graph
    :param n_iters: number of iteration to perform merging clusters
    :param min_comm_size: minimum number of nodes in a cluster
    :param max_comm_size: maximum number of nodes in a cluster
    :param verbose: if true => tracing progress by printing output
    :return: a list, whose element is a list of nodes that is in the same cluster after perform merging
    """
    parts = deepcopy(orig_parts)
    for _ in tqdm(range(n_iters), desc='Adding clusters together'):
        st_idx = 0
        while st_idx < len(parts):
            nd_idx = st_idx + 1
            idx_max = None
            max_ratio = None
            while nd_idx < len(parts):
                if len(parts[st_idx]) > min_comm_size:
                    break
                if len(parts[st_idx]) + len(parts[nd_idx]) > max_comm_size:
                    # if (len(parts[st_idx]) + len(parts[nd_idx]) > max_comm_size) and (len(parts[nd_idx]) > min_comm_size):
                    break
                cur_ratio = outer_edge_ratio(graph, parts[st_idx], parts[nd_idx])
                if max_ratio is None or max_ratio < cur_ratio:
                    max_ratio = cur_ratio
                    idx_max = nd_idx

                nd_idx += 1

            if max_ratio is not None:
                if verbose:
                    print(f"Merging cluster of size {len(parts[st_idx])} and {len(parts[idx_max])}")
                insort_right(parts, parts[st_idx] + parts[idx_max], key=len)
                del parts[idx_max]
                del parts[st_idx]
                st_idx -= 1
                if verbose:
                    print(f"Number of clusters remain: {len(parts)}")

            st_idx += 1

    return parts


def custom_clustering(graph, leiden_parts, max_comm_size=10000, min_comm_size=7500, n_iters=1, verbose=False,
                      overlap=0):
    """
    custom exist partitions generate by leiden algorithm, which large partition is divided and small partition is merge together
    :param graph: graph need to partition
    :param leiden_parts: clusters generated by leiden algorithm
    :param max_comm_size: maximum number of nodes for a cluster
    :param min_comm_size: minimum number of nodes for a cluster
    :param n_iters: number of iteration to perform merging clusters
    :param verbose: if true => tracing progress by printing output
    :return: a list, whose element is a list of nodes that is in the same cluster after perform custom clustering
    """
    parts = deepcopy(leiden_parts)
    overlap_list = []
    for it in range(n_iters):
        print(f"\n===============Iteration {it + 1}===============\n")
        # Continue partition cluster too large
        parts = leiden_recursive(graph, parts, max_comm_size, verbose)
        # Consider singleton communities as overlap community
        if overlap:
            overlap_list.extend(singleton_communities(parts, size=overlap))
        # Merge cluster too small to form larger cluster
        max_size = int(1.2 * max_comm_size) if it != n_iters - 1 else max_comm_size
        parts = merge_clusters(graph, parts, max_comm_size=max_size, min_comm_size=min_comm_size,
                               verbose=verbose)

    return parts, sorted(list(set(overlap_list)))


def node_species_histogram(data, clusters):
    histograms = []
    for cluster in clusters:
        cluster_species = data.node_species[cluster]
        histogram = np.histogram(cluster_species.numpy(), bins=[0]+(data.node_species.unique()+1).tolist())
        histograms.append(histogram[0])
    return np.array(histograms)


def histogram_plot(histograms, xticks, label="Histogram", color='g', xlabel="Node species"):
    fig, ax = plt.subplots(nrows=math.ceil(len(histograms)/4), ncols=4, figsize=(18, 8))
    ax = ax.flatten()

    for idx, histogram in enumerate(histograms):
        bar_plot = ax[idx].bar(torch.arange(len(xticks)), histogram, label=label, color=color)
        ax[idx].bar_label(bar_plot)
        ax[idx].set_xlabel(f"{xlabel}_Cluster_{idx+1}")
        ax[idx].set_ylabel("Count")
        ax[idx].set_xticks(torch.arange(len(xticks)), torch.arange(len(xticks)).numpy()+1)

    for idx in range(len(histograms), len(ax)):
        ax[idx].remove()

    fig.suptitle(f"{xlabel}_{label}", fontsize=20, weight='bold', color='b')
    plt.show(block=False)
    plt.pause(5)
    plt.close()


if __name__ == '__main__':
    dirname = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    dataset = PygNodePropPredDataset('ogbn-proteins', root='F:\Thesis\Graph_Clustering\dataset')
    G = build_graph(dirname)

    # Leiden cluster
    ig.summary(G)
    for cnt in range(1):
        # leiden_parts = leiden_clustering(G, None, verbose=True)
        leiden_parts = [dataset.get_idx_split()['test']]
        parts, overlap_list = custom_clustering(G, leiden_parts, max_comm_size=100, min_comm_size=50, n_iters=1,
                                                verbose=True,
                                                overlap=10)

        print(f"Number of clusters: {len(parts)}")
        for i, part in enumerate(parts):
            print(f"Cluster {i} contain {len(part)} nodes")
        cluster_writer(parts + [overlap_list], os.path.join(dirname, f'leiden_clusters_50_100_raw_test_overlap.csv'))
