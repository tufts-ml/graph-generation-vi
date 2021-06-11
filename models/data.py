import pickle
import torch
from torch.utils.data import Dataset
import dgl


class Graph_from_file(Dataset):
    # TODO implement dataset

    def __init__(self, args, graph_list, feature_map):
        print('Reading graphs from fiels...')
        self.dataset_path = args.current_processed_dataset_path
        self.graph_list = graph_list
        self.feature_map = feature_map

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        with open(self.dataset_path + 'graph' + str(self.graph_list[idx]) + '.dat', 'rb') as f:
            G = pickle.load(f)
        f.close()

        # TODO: prepare the data format required by gcn
        dgl_G = dgl.from_networkx(nx_graph=G)


        #set attribute for gcn
        n = len(G.nodes())
        len_node_vec = len(self.feature_map['node_forward'])
        node_mat = torch.zeros((n, 1), requires_grad=False)
        node_map = self.feature_map['node_forward']

        for v, data in G.nodes.data():
            ind = node_map[data['label']]

            node_mat[v, 0] = ind   #1?



        dgl_G.ndata['feat'] = node_mat

        return {"G": G, "dG": dgl_G}

    def collate_batch(self, batch):
        return batch