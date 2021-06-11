import os
import random
import time
import math
import pickle
from functools import partial
from multiprocessing import Pool
import bisect
import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from utils import mkdir, caveman_special, n_community, perturb_new, save_graphs
from datasets.preprocess import (
    mapping, random_walk_with_restart_sampling
)

def default_label_graph(G):
    for node in G.nodes():
        G.nodes[node]['label'] = 'DEFAULT_LABEL'
    for edge in G.edges():
        G.edges[edge]['label'] = 'DEFAULT_LABEL'
def check_graph_size(
    graph, min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):

    if min_num_nodes and graph.number_of_nodes() < min_num_nodes:
        return False
    if max_num_nodes and graph.number_of_nodes() > max_num_nodes:
        return False

    if min_num_edges and graph.number_of_edges() < min_num_edges:
        return False
    if max_num_edges and graph.number_of_edges() > max_num_edges:
        return False

    return True


def produce_graphs_from_raw_format(
    args, inputfile, output_path, num_graphs=None, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    """
    :param inputfile: Path to file containing graphs in raw format
    :param output_path: Path to store networkx graphs
    :param num_graphs: Upper bound on number of graphs to be taken
    :param min_num_nodes: Lower bound on number of nodes in graphs if provided
    :param max_num_nodes: Upper bound on number of nodes in graphs if provided
    :param min_num_edges: Lower bound on number of edges in graphs if provided
    :param max_num_edges: Upper bound on number of edges in graphs if provided
    :return: number of graphs produced
    """

    lines = []
    with open(inputfile, 'r') as fr:
        for line in fr:
            line = line.strip().split()
            lines.append(line)

    index = 0
    count = 0
    graphs_ids = set()
    while index < len(lines):
        if lines[index][0][1:] not in graphs_ids:
            graph_id = lines[index][0][1:]
            G = nx.Graph(id=graph_id)
            index += 1
            vert = int(lines[index][0])
            index += 1
            for i in range(vert):
                if args.label:
                    G.add_node(i, label=lines[index][0])
                else:
                    G.add_node(i, label='DEFAULT_LABEL')
                index += 1

            edges = int(lines[index][0])
            index += 1
            for i in range(edges):
                if args.label:
                    G.add_edge(int(lines[index][0]), int(
                        lines[index][1]), label=lines[index][2])
                else:
                    G.add_edge(int(lines[index][0]), int(
                        lines[index][1]), label='DEFAULT_LABEL')

                index += 1

            index += 1

            if not check_graph_size(
                G, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
            ):
                continue

            if nx.is_connected(G):
                with open(os.path.join(
                        output_path, 'graph{}.dat'.format(count)), 'wb') as f:
                    pickle.dump(G, f)

                graphs_ids.add(graph_id)
                count += 1

                if num_graphs and count >= num_graphs:
                    break

        else:
            vert = int(lines[index + 1][0])
            edges = int(lines[index + 2 + vert][0])
            index += vert + edges + 4

    return count


# For Enzymes, DD, Protetin dataset
def produce_graphs_from_graphrnn_format(
    args, input_path, dataset_name, output_path, num_graphs=None,
    node_invariants=[], min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):
    node_attributes = False
    graph_labels = False

    G = nx.Graph()
    # load data
    path = input_path
    data_adj = np.loadtxt(os.path.join(path, dataset_name + '_A.txt'),
                          delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(
            os.path.join(path, dataset_name + '_node_attributes.txt'),
            delimiter=',')

    data_node_label = np.loadtxt(
        os.path.join(path, dataset_name + '_node_labels.txt'),
        delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(
        os.path.join(path, dataset_name + '_graph_indicator.txt'),
        delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(
            os.path.join(path, dataset_name + '_graph_labels.txt'),
            delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)

    # add node labels
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
        if args.label:
            G.add_node(i + 1, label=str(data_node_label[i]))
        else:
            G.add_node(i+1, label='DEFAULT_LABEL')

    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1

    count = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['id'] = data_graph_labels[i]

        if not check_graph_size(
            G_sub, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
        ):
            continue

        if nx.is_connected(G_sub):
            G_sub = nx.convert_node_labels_to_integers(G_sub)
            G_sub.remove_edges_from(nx.selfloop_edges(G_sub))

            if 'CC' in node_invariants:
                clustering_coeff = nx.clustering(G_sub)
                cc_bins = [0, 0.2, 0.4, 0.6, 0.8]

            for node in G_sub.nodes():
                node_label = str(G_sub.nodes[node]['label'])

                if 'Degree' in node_invariants:
                    node_label += '-' + str(G_sub.degree[node])

                if 'CC' in node_invariants:
                    node_label += '-' + str(
                        bisect.bisect(cc_bins, clustering_coeff[node]))

               # G_sub.nodes[node]['label'] = node_label

            nx.set_edge_attributes(G_sub, 'DEFAULT_LABEL', 'label')

            with open(os.path.join(
                    output_path, 'graph{}.dat'.format(count)), 'wb') as f:
                pickle.dump(G_sub, f)

            count += 1

            if num_graphs and count >= num_graphs:
                break

    return count


def sample_subgraphs(
    args, idx, G, output_path, iterations, num_factor, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    count = 0
    deg = G.degree[idx]
    for _ in range(num_factor * int(math.sqrt(deg))):
        G_rw = random_walk_with_restart_sampling(
            args, G, idx, iterations=iterations, max_nodes=max_num_nodes,
            max_edges=max_num_edges)
        G_rw = nx.convert_node_labels_to_integers(G_rw)
        G_rw.remove_edges_from(nx.selfloop_edges(G_rw))

        if not check_graph_size(
            G_rw, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
        ):
            continue

        if nx.is_connected(G_rw):
            with open(os.path.join(
                    output_path,
                    'graph{}-{}.dat'.format(idx, count)), 'wb') as f:
                pickle.dump(G_rw, f)
                count += 1


def produce_random_walk_sampled_graphs(
    args, input_path, dataset_name, output_path, iterations, num_factor,
    num_graphs=None, min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):
    print('Producing random_walk graphs - num_factor - {}'.format(num_factor))
    G = nx.Graph()

    d = {}
    count = 0
    with open(os.path.join(input_path, dataset_name + '.content'), 'r') as f:
        for line in f.readlines():
            spp = line.strip().split('\t')
            if args.label:
                G.add_node(count, label=spp[-1])
            else:
                G.add_node(count, label='DEFAULT_LABEL')
            d[spp[0]] = count
            count += 1

    count = 0
    with open(os.path.join(input_path, dataset_name + '.cites'), 'r') as f:
        for line in f.readlines():
            spp = line.strip().split('\t')
            if spp[0] in d and spp[1] in d:
                G.add_edge(d[spp[0]], d[spp[1]], label='DEFAULT_LABEL')
            else:
                count += 1

    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G)

    with Pool(processes=48) as pool:
        for _ in tqdm(pool.imap_unordered(partial(
                sample_subgraphs, args, G=G, output_path=output_path,
                iterations=iterations, num_factor=num_factor,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges),
                list(range(G.number_of_nodes())))):
            pass

    filenames = []
    for name in os.listdir(output_path):
        if name.endswith('.dat'):
            filenames.append(name)

    random.shuffle(filenames)

    if not num_graphs:
        num_graphs = len(filenames)

    count = 0
    for i, name in enumerate(filenames[:num_graphs]):
        os.rename(
            os.path.join(output_path, name),
            os.path.join(output_path, 'graph{}.dat'.format(i))
        )
        count += 1

    for name in filenames[num_graphs:]:
        os.remove(os.path.join(output_path, name))

    return count


# Routine to create datasets
def create_graphs(args):
    base_path = os.path.join(args.dataset_path, f'{args.graph_type}/')
    # provide fake dataset (adopt from graphRNN code)
    if args.graph_type=='ladder':
        graphs = []
        for i in range(100, 201):
            graph = nx.ladder_graph(i)
            default_label_graph(graph)
            graphs.append(graph)
        args.max_prev_node = 10
    elif args.graph_type=='ladder_small':
        graphs = []
        for i in range(2, 11):
            graph = nx.ladder_graph(i)
            default_label_graph(graph)
            graphs.append(graph)
        args.max_prev_node = 10
    elif args.graph_type=='tree':
        graphs = []
        for i in range(2, 5):
            for j in range(3, 5):
                graph = nx.balanced_tree(i,j)
                default_label_graph(graph)
                graphs.append(graph)
        args.max_prev_node = 256
    elif args.graph_type=='caveman':
        graphs = []
        for i in range(2, 3):
            for j in range(30, 81):
                for k in range(10):
                    graph = caveman_special(i, j, p_edge=0.3)
                    default_label_graph(graph)
                    graphs.append(graph)
        args.max_prev_node = 100

    elif args.graph_type=='caveman_small':
        graphs = []
        for i in range(2, 3):
            for j in range(6, 11):
                for k in range(100):
                    graph = caveman_special(i, j, p_edge=0.8) # default 0.8
                    default_label_graph(graph)
                    graphs.append(graph)
        args.max_prev_node = 20

    elif args.graph_type=='path':
        graphs = []
        for l in range(2, 51):
            graph = nx.path_graph(l) # default 0.8
            default_label_graph(graph)
            graphs.append(graph)
        args.max_prev_node = 50

    elif args.graph_type=='caveman_small_single':
        graphs = []
        for i in range(2, 3):
            for j in range(8, 9):
                for k in range(100):
                    graph = caveman_special(i, j, p_edge=0.5)
                    default_label_graph(graph)
                    graphs.append(graph)
        args.max_prev_node = 20
    elif args.graph_type.startswith('community'):
        graphs = []
        num_communities = int(args.graph_type[-1])
        print('Creating dataset with ', num_communities, ' communities')

        # c_sizes = [15] * num_communities
        for k in range(3000):
            c_sizes = np.random.choice(np.arange(start=15,stop=30), num_communities)
            graph = n_community(c_sizes, p_inter=0.01)
            default_label_graph(graph)
            graphs.append(graph)
        args.max_prev_node = 80
    elif args.graph_type=='grid':
        graphs = []
        for i in range(10,20):
            for j in range(10,20):
                graph = nx.grid_2d_graph(i,j)
                default_label_graph(graph)
                graphs.append(graph)
        args.max_prev_node = 40
    elif args.graph_type=='grid_small':
        graphs = []
        for i in range(2,8):
            for j in range(2,8):
                graph = nx.grid_2d_graph(i, j)
                nodes = list(graph.nodes())
                node_mapping = {nodes[i]: i for i in range(len(nodes))}
                graph = nx.relabel_nodes(graph, node_mapping)
                default_label_graph(graph)
                graphs.append(graph)
        args.max_prev_node = 15
    elif args.graph_type=='grid_big':
        graphs = []
        for i in range(36, 46):
            for j in range(36, 46):
                graph = nx.grid_2d_graph(i,j)
                default_label_graph(graph)
                graphs.append(graph)
        args.max_prev_node = 90
    elif args.graph_type=='barabasi':
        graphs = []
        for i in range(100,200):
             for j in range(4,5):
                 for k in range(5):
                     graph = nx.barabasi_albert_graph(i,j)
                     default_label_graph(graph)
                     graphs.append(graph)
        args.max_prev_node = 130
    elif args.graph_type=='barabasi_small':
        graphs = []
        for i in range(4,21):
             for j in range(3,4):
                 for k in range(10):
                     graph = nx.barabasi_albert_graph(i,j)
                     default_label_graph(graph)
                     graphs.append(graph)
        args.max_prev_node = 20

    elif 'barabasi_noise' in args.graph_type:
        graphs = []
        for i in range(100,101):
            for j in range(4,5):
                for k in range(500):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        graphs = perturb_new(graphs, p=args.noise/10.0)
        graphs = [default_label_graph(graph) for graph in graphs]
        args.max_prev_node = 99

    # real  datasets
    elif 'PROTEINS_full' == args.graph_type:
        node_invariants = ['Degree']
        min_num_nodes, max_num_nodes = 20, None
        min_num_edges, max_num_edges = None, None
        args.max_prev_node = 80

    elif 'DD' == args.graph_type:
        node_invariants = ['Degree']
        min_num_nodes, max_num_nodes = 100, 500
        min_num_edges, max_num_edges = None, None
        args.max_prev_node = 230

    elif 'ENZYMES' in args.graph_type:
        # Node invariants - Options 'Degree' and 'CC'
        node_invariants = ['Degree']
        min_num_nodes, max_num_nodes = 10, None
        min_num_edges, max_num_edges = None, None
        args.max_prev_node = 25

    elif 'Lung' == args.graph_type:
        # base_path = os.path.join(args.dataset_path, 'Lung/')
        input_path = base_path + 'lung.txt'
        min_num_nodes, max_num_nodes = None, 50
        min_num_edges, max_num_edges = None, None
        args.max_prev_node = 47

    elif 'Breast' == args.graph_type:
        # base_path = os.path.join(args.dataset_path, 'Breast/')
        input_path = base_path + 'breast.txt'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None
        args.max_prev_node = 90

    elif 'Leukemia' == args.graph_type:
        # base_path = os.path.join(args.dataset_path, 'Leukemia/')
        input_path = base_path + 'leukemia.txt'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None
        args.max_prev_node = 89

    elif 'Yeast' == args.graph_type:
        # base_path = os.path.join(args.dataset_path, 'Yeast/')
        input_path = base_path + 'yeast.txt'
        min_num_nodes, max_num_nodes = None, 50
        min_num_edges, max_num_edges = None, None
        args.max_prev_node = 47

    elif 'All' == args.graph_type:
        # base_path = os.path.join(args.dataset_path, 'All/')
        input_path = base_path + 'all.txt'
        # No limit on number of nodes and edges
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'citeseer' == args.graph_type:
        # base_path = os.path.join(args.dataset_path, 'citeseer/')
        random_walk_iterations = 150  # Controls size of graph
        num_factor = 5  # Controls size of dataset
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = 20, None

    elif 'citeseer_small' == args.graph_type:
        # base_path = os.path.join(args.dataset_path, 'citeseer_small/')
        random_walk_iterations = 150  # Controls size of graph
        num_factor = 5  # Controls size of dataset
        min_num_nodes, max_num_nodes = 4, 20
        min_num_edges, max_num_edges = None, None

    elif 'cora' in args.graph_type:
        # base_path = os.path.join(args.dataset_path, 'cora/')
        random_walk_iterations = 150  # Controls size of graph
        num_factor = 5  # Controls size of dataset

        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = 20, None

    else:
        print('Dataset - {} is not valid'.format(args.graph_type))
        exit()
    args.current_dataset_path = os.path.join(base_path, 'graphs/')
    args.current_processed_dataset_path = args.current_dataset_path


    if args.produce_graphs:
        mkdir(args.current_dataset_path)

        if args.graph_type in ['Lung', 'Breast', 'Leukemia', 'Yeast', 'All']:
            count = produce_graphs_from_raw_format(
                args, input_path, args.current_dataset_path, args.num_graphs,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        elif args.graph_type in ['ENZYMES', 'DD', 'PROTEINS_full']:
            count = produce_graphs_from_graphrnn_format(
                args, base_path, args.graph_type, args.current_dataset_path,
                num_graphs=args.num_graphs, node_invariants=node_invariants,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        elif args.graph_type in ['cora', 'citeseer_small', 'citeseer']:
            count = produce_random_walk_sampled_graphs(
                args, base_path, args.graph_type, args.current_dataset_path,
                num_graphs=args.num_graphs, iterations=random_walk_iterations,
                num_factor=num_factor, min_num_nodes=min_num_nodes,
                max_num_nodes=max_num_nodes, min_num_edges=min_num_edges,
                max_num_edges=max_num_edges)

        elif args.graph_type in ['ladder', 'ladder_small', 'tree', 'caveman',
                                'caveman_small', 'caveman_small_single', 'grid',
                                'grid_small', 'barabasi', 'barabasi_small', 'grid_big', 'path'] or\
            args.graph_type.startswith('community') or 'barabasi_noise' in args.graph_type:
            save_graphs(args.current_dataset_path, graphs)
            count = len(graphs)

        print('Graphs produced', count)
    else:
        try:
            count = len([name for name in os.listdir(
                args.current_dataset_path) if name.endswith(".dat")])
            print('Graphs counted', count)
        except:
            print(f'no {args.current_dataset_path}, please generate graphs first')

    # Produce feature map
    feature_map = mapping(args.current_dataset_path,
                          args.current_dataset_path + 'map.dict')
    print(feature_map)

    graphs = [i for i in range(count)]
    random.seed(32)
    if args.data_small == True and len(graphs) > 400:
        print('Using small dataset....')
        graphs = random.sample(graphs, 400)



    return graphs
