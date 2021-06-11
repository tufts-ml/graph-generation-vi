import pickle
import os
from multiprocessing import Pool
from functools import partial
import networkx as nx
import torch
from tqdm.auto import tqdm
import subprocess
import tempfile
MAX_WORKERS = 48


def mapping(path, dest):
    """
    :param path: path to folder which contains pickled networkx graphs
    :param dest: place where final dictionary pickle file is stored
    :return: dictionary of 4 dictionary which contains forward 
    and backwards mappings of vertices and labels, max_nodes and max_edges
    """

    node_forward, node_backward = {}, {}
    edge_forward, edge_backward = {}, {}
    node_count, edge_count = 0, 0
    max_nodes, max_edges, max_degree = 0, 0, 0
    min_nodes, min_edges = float('inf'), float('inf')
    total_node = 0
    for filename in tqdm(os.listdir(path)):
        if filename.endswith(".dat"):
            f = open(path + filename, 'rb')
            G = pickle.load(f)
            f.close()
            max_nodes = max(max_nodes, len(G.nodes()))
            min_nodes = min(min_nodes, len(G.nodes()))
            for _, data in G.nodes.data():
                if data['label'] not in node_forward:
                    node_forward[data['label']] = node_count
                    node_backward[node_count] = data['label']
                    node_count += 1

            max_edges = max(max_edges, len(G.edges()))
            min_edges = min(min_edges, len(G.edges()))
            for _, _, data in G.edges.data():
                if data['label'] not in edge_forward:
                    edge_forward[data['label']] = edge_count
                    edge_backward[edge_count] = data['label']
                    edge_count += 1

            max_degree = max(max_degree, max([d for n, d in G.degree()]))

    feature_map = {
        'node_forward': node_forward,
        'node_backward': node_backward,
        'edge_forward': edge_forward,
        'edge_backward': edge_backward,
        'max_nodes': max_nodes,
        'min_nodes': min_nodes,
        'max_edges': max_edges,
        'min_edges': min_edges,
        'max_degree': max_degree
    }

    f = open(dest, 'wb')
    pickle.dump(feature_map, f)
    f.close()

    print('Successfully done node count', node_count)
    print('Successfully done edge count', edge_count)

    return feature_map


def get_bfs_seq(G, start_id):
    """
    Get a bfs node sequence
    :param G: graph
    :param start_id: starting node
    :return: List of bfs node sequence
    """
    successors_dict = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        succ = []
        for current in start:
            if current in successors_dict:
                succ = succ + successors_dict[current]

        output = output + succ
        start = succ
    return output


def get_random_bfs_seq(graph):
    n = len(graph.nodes())
    # Create a random permutaion of graph nodes
    perm = torch.randperm(n)
    adj = nx.to_numpy_matrix(graph, nodelist=perm.numpy().tolist(), dtype=int)
    G = nx.from_numpy_matrix(adj)

    # Construct bfs ordering starting from a random node
    start_id = 0
    bfs_seq = get_bfs_seq(G, start_id)

    return [perm[bfs_seq[i]] for i in range(n)], perm


def calc_max_prev_node_helper(idx, graphs_path):
    with open(graphs_path + 'graph' + str(idx) + '.dat', 'rb') as f:
        G = pickle.load(f)

    max_prev_node = []
    for _ in range(100):
        bfs_seq, _ = get_random_bfs_seq(G)
        bfs_order_map = {bfs_seq[i]: i for i in range(len(G.nodes()))}
        G = nx.relabel_nodes(G, bfs_order_map)

        max_prev_node_iter = 0
        for u, v in G.edges():
            max_prev_node_iter = max(max_prev_node_iter, max(u, v) - min(u, v))

        max_prev_node.append(max_prev_node_iter)

    return max_prev_node


def calc_max_prev_node(graphs_path):
    """
    Approximate max_prev_node from simulating bfs sequences 
    """
    max_prev_node = []
    count = len([name for name in os.listdir(
        graphs_path) if name.endswith(".dat")])

    max_prev_node = []
    with Pool(processes=8) as pool:
        for max_prev_node_g in tqdm(pool.imap_unordered(
                partial(calc_max_prev_node_helper, graphs_path=graphs_path), list(range(count)))):
            max_prev_node.extend(max_prev_node_g)

    max_prev_node = sorted(max_prev_node)[-1 * int(0.001 * len(max_prev_node))]
    return max_prev_node


def random_walk_with_restart_sampling(
    args, G, start_node, iterations, fly_back_prob=0.15,
    max_nodes=None, max_edges=None
):
    sampled_graph = nx.Graph()
    if args.label:
        sampled_graph.add_node(start_node, label=G.nodes[start_node]['label'])
    else:
        sampled_graph.add_node(start_node, label='DEFAULT_LABEL')

    curr_node = start_node

    for _ in range(iterations):
        choice = torch.rand(()).item()

        if choice < fly_back_prob:
            curr_node = start_node
        else:
            neigh = list(G.neighbors(curr_node))
            chosen_node_id = torch.randint(
                0, len(neigh), ()).item()
            chosen_node = neigh[chosen_node_id]
            if args.label:
                sampled_graph.add_node(
                    chosen_node, label=G.nodes[chosen_node]['label'])
                sampled_graph.add_edge(
                    curr_node, chosen_node, label=G.edges[curr_node, chosen_node]['label'])
            else:
                sampled_graph.add_node(
                    chosen_node, label='DEFAULT_LABEL')
                sampled_graph.add_edge(
                    curr_node, chosen_node, label='DEFAULT_LABEL')


            curr_node = chosen_node

        if max_nodes is not None and sampled_graph.number_of_nodes() >= max_nodes:
            break

        if max_edges is not None and sampled_graph.number_of_edges() >= max_edges:
            break

    # sampled_graph = G.subgraph(sampled_node_set)

    return sampled_graph

def get_min_dfscode(G, temp_path=tempfile.gettempdir()):
    input_fd, input_path = tempfile.mkstemp(dir=temp_path)

    with open(input_path, 'w') as f:
        vcount = len(G.nodes)
        f.write(str(vcount) + '\n')
        i = 0
        d = {}
        for x in G.nodes:
            d[x] = i
            i += 1
            f.write(str(G.nodes[x]['label']) + '\n')

        ecount = len(G.edges)
        f.write(str(ecount) + '\n')
        for (u, v) in G.edges:
            f.write(str(d[u]) + ' ' + str(d[v]) +
                    ' ' + str(G[u][v]['label']) + '\n')

    output_fd, output_path = tempfile.mkstemp(dir=temp_path)

    dfscode_bin_path = 'bin/dfscode'
    with open(input_path, 'r') as f:
        subprocess.call([dfscode_bin_path, output_path, '2'], stdin=f)

    with open(output_path, 'r') as dfsfile:
        dfs_sequence = []
        for row in dfsfile.readlines():
            splited_row = row.split()
            splited_row = [splited_row[2 * i + 1] for i in range(5)]
            dfs_sequence.append(splited_row)

    os.close(input_fd)
    os.close(output_fd)

    try:
        os.remove(input_path)
        os.remove(output_path)
    except OSError:
        pass

    return dfs_sequence