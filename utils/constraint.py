import numpy as np
from bisect import insort_right
from .cluster import (
    leiden_clustering,
    singleton_communities,
    custom_clustering,
    build_graph,
    cluster_writer,
)
import os
from ogb.nodeproppred import PygNodePropPredDataset
from copy import deepcopy
import igraph as ig


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


# Use for minimum constraint
def merge_small_to_large(graph, small_part, large_parts, max_comm_size=500, verbose=False):
    """
    The Leiden algorithm with minimum constraint. Every small communities (small_part) is considered to merge
    with larger community (large_parts), which have the largest edge ratio when small communities move to the
    same community of the larger one
    :param graph: original graph, which contain all node and edges.
    :param small_part: a list of communities whose size is consider to be small
    :param large_parts: a list of communities whose size is consider to be large
    :param max_comm_size: A merging is performed only if total number of node is not exceed max_comm_size
    :param verbose: using to trace the process. If true, print state whenever a merging is performed
    :return:
    """
    max_ratio = []
    idx_list = []

    for large_idx in range(len(large_parts)):
        if len(small_part) + len(large_parts[large_idx]) > max_comm_size:
            break
        cur_ratio = outer_edge_ratio(graph, small_part, large_parts[large_idx], eps=1e-5)
        max_ratio.append(cur_ratio)
        idx_list.append(large_idx)

    if max_ratio:
        max_ratio = np.array(max_ratio)
        max_ratio = max_ratio / np.sum(max_ratio)

        idx = np.random.choice(np.arange(len(idx_list)), size=1, replace=False, p=max_ratio)[0]

        idx_max = idx_list[idx]
        if verbose:
            print(f"Merging cluster of size {len(small_part)} and {len(large_parts[idx_max])}")
        insort_right(large_parts, small_part + large_parts[idx_max], key=len)
        del large_parts[idx_max]

        return large_parts, True
    else:
        if verbose:
            print("Cannot merging clusters together")
        return large_parts, False


# Use for maximum constraint
def leiden_reserve_node_number(graph, part, verbose=True, max_comm_size=200):
    """
    Using the leiden algorithm recursively until there is no community whose size is bigger than max_comm_size
    :param graph: ogbn-proteins dataset
    :param part: a list of original community extracted by the leiden algorithm
    :param verbose: If true, print state of the leiden algorithm
    :param max_comm_size: maximum size of each community
    :return: a list of community
    """
    subG = graph.subgraph(part)
    if verbose:
        print("Start clustering")
    sub_parts = leiden_clustering(subG, verbose=verbose)
    if len(sub_parts) == 1:
        print(f"\033[93mWarning: Divided again\033[0m")
        sub_parts = leiden_clustering(subG, verbose=True, max_comm_size=max_comm_size)
    sub_parts = [[subG.vs[p]['name'] for p in P] for P in sub_parts]

    return sub_parts


# Min/Max community size constraint
def merge_with_min_comm(graph, orig_clusters, min_comm_size=400, max_comm_size=500, n_cluster=20, max_iters=1e4):
    """
    The leiden algorithm with minimum and maximum community size constraint
    :param graph: ogbn-proteins dataset
    :param orig_clusters: original communities extracted by the leiden algorithm
    :param min_comm_size: minimum community size constraint
    :param max_comm_size: maximum community size constraint
    :param n_cluster: Avoid redundant community. The number of output communities need to be the multiples of n_cluster
    :param max_iters: the maximum number of iteration used to perform the leiden algorithm
    :return: a list of communities
    """
    proc_clusters = deepcopy(orig_clusters)
    it = 0
    overlap_list = set()
    while (len(proc_clusters) % n_cluster) or (len(proc_clusters[0]) < min_comm_size):
        if it == 0:
            print("Starting...")
        if it > max_iters:
            break
        print(f"\nNumber of clusters remain: {len(proc_clusters)}")
        print(f"Smallest cluster: {len(proc_clusters[0])}")

        small_clusters = leiden_reserve_node_number(graph, proc_clusters[0], max_comm_size=max_comm_size)
        if len(small_clusters) == 1:
            small_clusters = leiden_reserve_node_number(graph, proc_clusters[0],
                                                        max_comm_size=(max_comm_size - min_comm_size) // 2)
        proc_clusters = proc_clusters[1:]

        overlap_list.update(singleton_communities(small_clusters, size=5))

        remain_smalls = []
        for small_cluster in small_clusters:
            proc_clusters, flag = merge_small_to_large(graph, small_cluster, proc_clusters, max_comm_size=max_comm_size,
                                                       verbose=True)
            if not flag:
                remain_smalls.append(small_cluster)
        proc_clusters = sorted(remain_smalls, key=len) + proc_clusters
        it += 1

    return proc_clusters, sorted(list(overlap_list))


if __name__ == '__main__':
    dirname = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    dataset = PygNodePropPredDataset('ogbn-proteins', root=dirname)
    G = build_graph(dirname)

    # Leiden cluster with min/max community size constraint
    ig.summary(G)

    leiden_parts = leiden_clustering(G, None, verbose=True)
    parts, _ = custom_clustering(G, leiden_parts, max_comm_size=100, min_comm_size=50, n_iters=1,
                                 verbose=True,
                                 overlap=0)
    print(f"Number of clusters: {len(parts)}")
    for i, part in enumerate(parts):
        print(f"Cluster {i} contain {len(part)} nodes")
    cluster_writer(parts, os.path.join(dirname, f'leiden_clusters_test_raw.csv'))

    parts, _ = merge_with_min_comm(G, parts, min_comm_size=80, max_comm_size=100, n_cluster=20)
    cluster_writer(parts, os.path.join(dirname, f'leiden_clusters_test_final.csv'))





