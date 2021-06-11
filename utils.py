import os
import shutil
import pickle
import torch
import networkx as nx
import pynauty as pnt
import numpy as np


def mkdir(path):
    if os.path.isdir(path):
        is_del = input('Delete ' + path + ' Y/N:')
        if is_del.strip().lower() == 'y':
            shutil.rmtree(path)
        else:
            exit()

    os.makedirs(path)


def load_graphs(graphs_path, graphs_indices=None):
    """
    Returns a list of graphs given graphs directory and graph indices (Optional)
    If graphs_indices are not provided all graphs will be loaded
    """

    graphs = []
    if graphs_indices is None:
        for name in os.listdir(graphs_path):
            if not name.endswith('.dat'):
                continue

            with open(graphs_path + name, 'rb') as f:
                graphs.append(pickle.load(f))
    else:
        for ind in graphs_indices:
            with open(graphs_path + 'graph' + str(ind) + '.dat', 'rb') as f:
                graphs.append(pickle.load(f))

    return graphs


def save_graphs(graphs_path, graphs):
    """
    Save networkx graphs to a directory with indexing starting from 0
    """
    for i in range(len(graphs)):
        with open(graphs_path + 'graph' + str(i) + '.dat', 'wb') as f:
            pickle.dump(graphs[i], f)


# Create Directories for outputs
def create_dirs(args):
    if args.clean_tensorboard and os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)

    if args.clean_temp and os.path.isdir(args.temp_path):
        shutil.rmtree(args.temp_path)

    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)

    if not os.path.isdir(args.temp_path):
        os.makedirs(args.temp_path)

    if not os.path.isdir(args.tensorboard_path):
        os.makedirs(args.tensorboard_path)

    if not os.path.isdir(args.current_temp_path):
        os.makedirs(args.current_temp_path)

    if not os.path.isdir(args.logging_path):
        os.makedirs(args.logging_path)


def save_model(epoch, args, model, gcn=None, optimizer=None, scheduler=None, **extra_args):
    if not os.path.isdir(args.current_model_save_path):
        os.makedirs(args.current_model_save_path)

    fname = args.current_model_save_path +'epoch' + '_' + str(epoch) + '.dat'
    checkpoint = {'saved_args': args, 'epoch': epoch}

    save_items = {'model': model}
    if gcn is not None:
        save_items['gcn'] = {'gcn': gcn}

    if optimizer:
        save_items['optimizer'] = optimizer
    if scheduler:
        save_items['scheduler'] = scheduler

    for name, d in save_items.items():
        save_dict = {}
        for key, value in d.items():
            save_dict[key] = value.state_dict()

        checkpoint[name] = save_dict

    if extra_args:
        for arg_name, arg in extra_args.items():
            checkpoint[arg_name] = arg

    torch.save(checkpoint, fname)


def load_model(path, device, model, gcn=None, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location=device)

    for name, d in {'model': model, 'gcn': {'gcn': gcn}, 'optimizer': optimizer, 'scheduler': scheduler}.items():
        if d is not None:
            for key, value in d.items():
                if value is not None:
                    value.load_state_dict(checkpoint[name][key])

        if name == 'model' or name == 'gcn':
            for _, value in d.items():
                if value is not None:
                    value.to(device=device)


def get_last_checkpoint(args, epoch):
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = args.load_model_path + '/model_save/'
    # Checkpoint file names are in lexicographic order
    last_checkpoint_name = checkpoint_dir + 'epoch' + '_' + str(epoch) + '.dat'
    print('Last checkpoint is {}'.format(last_checkpoint_name))
    return last_checkpoint_name, epoch


def get_model_attribute(attribute, fname, device):

    checkpoint = torch.load(fname, map_location=device)

    return checkpoint[attribute]


def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)


def caveman_special(c=2,k=20,p_path=0.1,p_edge=0.3):
    p = p_path
    path_count = max(int(np.ceil(p * k)),1)
    G = nx.caveman_graph(c, k)
    # remove 50% edges
    p = 1-p_edge
    for (u, v) in list(G.edges()):
        if np.random.rand() < p and ((u < k and v < k) or (u >= k and v >= k)):
            G.remove_edge(u, v)
    # add path_count links
    for i in range(path_count):
        u = np.random.randint(0, k)
        v = np.random.randint(k, k * 2)
        G.add_edge(u, v)
    G = max(connected_component_subgraphs(G), key=len)
    return G


def n_community(c_sizes, p_inter=0.01):
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = list(connected_component_subgraphs(G))
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i+1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
    #print('connected comp: ', len(list(nx.connected_component_subgraphs(G))))
    return G


def perturb_new(graph_list, p):
    ''' Perturb the list of graphs by adding/removing edges.
    Args:
        p_add: probability of adding edges. If None, estimate it according to graph density,
            such that the expected number of added edges is equal to that of deleted edges.
        p_del: probability of removing edges
    Returns:
        A list of graphs that are perturbed from the original graphs
    '''
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_remove_count = 0
        for (u, v) in list(G.edges()):
            if np.random.rand()<p:
                G.remove_edge(u, v)
                edge_remove_count += 1
        # randomly add the edges back
        for i in range(edge_remove_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u,v)) and (u!=v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


def nx_to_nauty(nx_G):
    na_G = pnt.Graph(nx_G.number_of_nodes())
    for n_node in range(nx_G.number_of_nodes()):
        na_G.connect_vertex(n_node, list(nx_G.neighbors(n_node)))
    return na_G

def nauty_to_nx(na_G):
    raise NotImplementedError
    # pass

# if __name__ == '__main__':
#     import pynauty as pnt
#     import networkx as nx
#     g = nx.complete_graph(5)
#     na_g = nx_to_nauty(g)
#     print(pnt.autgrp(na_g))
#     g = nx.star_graph(4)
#     na_g = nx_to_nauty(g)
#     print(pnt.autgrp(na_g))


