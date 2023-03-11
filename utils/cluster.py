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


def generate_mini_batch(parts, num_cluster):
    part_per_batch = len(parts) // num_cluster

    random_idx = np.arange(len(parts))
    np.random.shuffle(random_idx)
    mini_batches = []
    for i in range(num_cluster - 1):
        mini_batch = []
        for parts_idx in range(i * part_per_batch, (i + 1) * part_per_batch):
            mini_batch.extend(parts[random_idx[parts_idx]])
        mini_batches.append(mini_batch)

    last_batch = []
    for parts_idx in range((num_cluster - 1) * part_per_batch, len(parts)):
        last_batch.extend(parts[random_idx[parts_idx]])
    mini_batches.append(last_batch)

    return mini_batches


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


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


if __name__ == '__main__':
    dirname = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    G = build_graph(dirname)

    # Leiden cluster
    ig.summary(G)
    for cnt in range(1):
        leiden_parts = leiden_clustering(G, None, verbose=True)
        parts, overlap_list = custom_clustering(G, leiden_parts, max_comm_size=100, min_comm_size=50, n_iters=1,
                                                verbose=True,
                                                overlap=3)

        print(f"Number of clusters: {len(parts)}")
        for i, part in enumerate(parts):
            print(f"Cluster {i} contain {len(part)} nodes")
        cluster_writer(parts+overlap_list, os.path.join(dirname, f'leiden_clusters_50_100_raw_weight_overlap.csv'))

    # Random cluster
    # parts = random_partition_graph(G.vcount(), cluster_number=20)
    # cluster_writer(parts, os.path.join(dirname, 'clusters_small.csv'))
