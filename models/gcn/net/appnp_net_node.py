"""APPNP and PPNP layers."""

import math
import numpy as np
import torch
from dgl.nn.pytorch import APPNPConv
from models.gcn.layer.mlp_readout_layer import MLPReadout
from torch import nn

class APPNET(nn.Module):

    def __init__(self, args, node_len, out_dim):
        super().__init__()
        in_dim_node = node_len  # node_dim (feat is an integer)
        hidden_dim = args.gcn_hidden_dim
        out_dim = out_dim
        in_feat_dropout = args.gcn_in_feat_dropout

        self.batch_norm = args.gcn_batch_norm
        self.residual = args.gcn_residual
        self.n_classes = args.gcn_n_classes
        self.device = args.device
        self.d_model = hidden_dim

        self.dropout = nn.Dropout(args.gcn_dropout)
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)  # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.APPNP = APPNPConv(10, alpha=0.05, edge_drop=0.2)
        self.MLP_layer = MLPReadout(out_dim, out_dim)

    def positionalencoding(self, lengths, permutations):
        # length = sum([len(perm) for perm in permutations])
        l_t = len(permutations[0])
        # pes = [torch.zeros(length, self.d_model) for length in lengths]
        pes = torch.split(torch.zeros((sum(lengths), self.d_model), device=self.device), lengths)
        # print(pes[0].device)
        position = torch.arange(0, l_t, device=self.device).unsqueeze(1) + 1
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float, device=self.device) *
                              -(math.log(10000.0) / self.d_model)))
        # test = torch.sin(position.float() * div_term)
        for i in range(len(lengths)):
            pes[i][permutations[i], 0::2] = torch.sin(position.float() * div_term)
            pes[i][permutations[i], 1::2] = torch.cos(position.float() * div_term)

        pes = torch.cat(pes)
        return pes

    def forward(self, g, h, p=None):
        # p: positional(order) embedding
        # input embedding

        h = self.embedding_h(torch.squeeze(h))
        if p is not None:
            p = self.positionalencoding(g.batch_num_nodes().tolist(), p)
            h = h + p

        h = self.dropout(h)
        # GCN
        # for conv in self.layers:
        #     h = conv(g, h)

        # APPNP
        h = self.APPNP(g, h)

        # output
        h_out = self.MLP_layer(h)
        # h_out = F.softmax(h_out, dim=0)
        return torch.squeeze(h_out)
# def normalize_adjacency_matrix(A, I):
#     """
#     Creating a normalized adjacency matrix with self loops.
#     :param A: Sparse adjacency matrix.
#     :param I: Identity matrix.
#     :return A_tile_hat: Normalized adjacency matrix.
#     """
#     A_tilde = A + I
#     degrees = A_tilde.sum(axis=0)[0].tolist()
#     D = sparse.diags(degrees, [0])
#     D = D.power(-0.5)
#     A_tilde_hat = D.dot(A_tilde).dot(D)
#     return A_tilde_hat
#
#
# def uniform(size, tensor):
#     """
#     Uniform weight initialization.
#     :param size: Size of the tensor.
#     :param tensor: Tensor initialized.
#     """
#     stdv = 1.0 / math.sqrt(size)
#     if tensor is not None:
#         tensor.data.uniform_(-stdv, stdv)
#
#
# class DenseFullyConnected(torch.nn.Module):
#     """
#     Abstract class for PageRank and Approximate PageRank networks.
#     :param in_channels: Number of input channels.
#     :param out_channels: Number of output channels.
#     :param density: Feature matrix structure.
#     """
#     def __init__(self, in_channels, out_channels):
#         super(DenseFullyConnected, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.define_parameters()
#         self.init_parameters()
#
#     def define_parameters(self):
#         """
#         Defining the weight matrices.
#         """
#         self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
#         self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))
#
#     def init_parameters(self):
#         """
#         Initializing weights.
#         """
#         torch.nn.init.xavier_uniform_(self.weight_matrix)
#         uniform(self.out_channels, self.bias)
#
#     def forward(self, features):
#         """
#         Doing a forward pass.
#         :param features: Feature matrix.
#         :return filtered_features: Convolved features.
#         """
#         filtered_features = torch.mm(features, self.weight_matrix)
#         filtered_features = filtered_features + self.bias
#         return filtered_features
#
#
# class APPNET(nn.Module):
#     def __init__(self, args, node_len, out_dim):
#         super().__init__()
#         in_dim_node = node_len  # node_dim (feat is an integer)
#         hidden_dim = args.gcn_hidden_dim
#         in_feat_dropout = args.gcn_in_feat_dropout
#         n_layers = args.gcn_num_layers
#         self.dropout = args.gcn_dropout
#         self.iterations = 5
#
#         self.batch_norm = args.gcn_batch_norm
#         self.residual = args.gcn_residual
#         self.n_classes = args.gcn_n_classes
#         self.device = args.device
#         self.d_model = hidden_dim
#         self.alpha = 0.05
#
#         self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)  # node feat is an integer
#         self.layer_2 = DenseFullyConnected(hidden_dim, out_dim)
#         self.Dropout = nn.Dropout(self.dropout)
#
#
#     def setup_propagator(self, A):
#         # A = create_adjacency_matrix(graph)
#         I = sparse.eye(A.shape[0])
#         A_tilde_hat = normalize_adjacency_matrix(A, I)
#         propagator = dict()
#         A_tilde_hat = sparse.coo_matrix(A_tilde_hat)
#         indices = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1).T
#         propagator["indices"] = torch.LongTensor(indices)
#         propagator["values"] = torch.FloatTensor(A_tilde_hat.data)
#         self.edge_indices = propagator["indices"].to(self.device)
#         self.edge_weights = propagator["values"].to(self.device)
#
#
#     def forward(self, feature):
#         # feature_values = torch.nn.functional.dropout(feature_values, p=self.args.dropout, training=self.training)
#         latent_features_1 = self.embedding_h(torch.squeeze(feature))
#
#         latent_features_1 = torch.nn.functional.relu(latent_features_1)
#
#         latent_features_1 = self.Dropout(latent_features_1)
#
#         latent_features_2 = self.layer_2(latent_features_1)
#         localized_predictions = latent_features_2
#
#         # edge_weights = self.Dropout(self.edge_weights)
#
#         for iteration in range(self.iterations):
#             new_features = spmm(index=self.edge_indices,
#                                 value=self.edge_weights,
#                                 n=localized_predictions.shape[0],
#                                 m=localized_predictions.shape[0],
#                                 matrix=localized_predictions)
#
#             localized_predictions = (1 - self.alpha) * new_features
#             localized_predictions = localized_predictions + self.alpha * latent_features_2
#
#         predictions = localized_predictions
#         return predictions




# if __name__ == '__main__':
#     graph = nx.caveman_graph(4, 3)
#     propagator = create_propagator_matrix(graph, 0.05, "approx")
#     edge_indices = propagator["indices"]
#     edge_weights = propagator["values"]
#
#     H = torch.rand((graph.number_of_nodes(), 4))
#     Z1 = H
#     n_iter = 10
#     for i in range(n_iter):
#         new_features = spmm(index=edge_indices,
#              value=edge_weights,
#              n=Z1.shape[0],
#              m=Z1.shape[0],
#              matrix=Z1)
#         Z1 = (1 - 0.05) * new_features
#         Z1 = Z1 + 0.05 * H
#     print(Z1)
#
#     I = torch.eye(graph.number_of_nodes())
#     FT = torch.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
#     for i in range(n_iter):
#         FT += I*0.05
#         I = spmm(index=edge_indices,
#              value=edge_weights,
#              n=I.shape[0],
#              m=I.shape[0],
#              matrix=I)
#
#         I *= (1-0.05)
#     fixed = I + FT
#     fixed = sparse.coo_matrix(fixed)
#     indices = np.concatenate([fixed.row.reshape(-1, 1), fixed.col.reshape(-1, 1)], axis=1).T
#     edge_indices_v2 = torch.LongTensor(indices)
#     edge_weights_v2 = torch.FloatTensor(fixed.data)
#     Z2 = spmm(index=edge_indices_v2,
#              value=edge_weights_v2,
#              n=H.shape[0],
#              m=H.shape[0],
#              matrix=H)
#     print(Z2)



