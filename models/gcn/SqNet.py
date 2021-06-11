import torch
import dgl
from torch import nn
from torch.nn import functional as F
from models.gcn.net.appnp_net_node import APPNET
from models.gcn.net.gat_net_node import GATNet
from models.gcn.net.gcn_net_node import GCNNet
from models.gcn.layer.mlp_readout_layer import MLPReadout
from models.gcn.helper import mp_sampler, compute_autogrp_n


class sequential_net0(nn.Module):
    def __init__(self, args, len_node_vec, hidden_dim=32, vf2=False):
        super().__init__()
        if args.gcn_type == 'gcn':
            self.node_embedding = GCNNet(args, len_node_vec, out_dim=hidden_dim).to(args.device)
        elif args.gcn_type == 'gat':
            self.node_embedding = GATNet(args, len_node_vec, out_dim=hidden_dim).to(args.device)
        elif args.gcn_type == 'appnp':
            self.node_embedding = APPNET(args, len_node_vec, out_dim=hidden_dim).to(args.device)

        self.node_readout = MLPReadout(hidden_dim, 1).to(args.device)
        self.args = args
        self.rep_computer = mp_sampler(args, vf2)
        self.sample_time = 0

    def forward(self, g, m=1):
        # m: sample_size
        unbatch_dG = [graph['dG'].to(self.args.device) for graph in g]
        batch_dG = dgl.batch(unbatch_dG)
        batch_dGX = batch_dG.ndata['feat'].long()
        batch_num_nodes = batch_dG.batch_num_nodes()

        # gcn forward
        node_embedds = self.node_embedding(batch_dG, batch_dGX)
        parameterizations = self.node_readout(node_embedds).squeeze()
        parameterizations = torch.split(parameterizations, batch_num_nodes.tolist())
        # print(parameterizations)
        log_reps = torch.empty((len(g), m), requires_grad=False)  # (N, M)
        log_probs = torch.empty((len(g), m), device=self.args.device)
        batch_perms = [[] for _ in range(m)]
        for idx_g in range(len(g)):
            # sample permutation
            batch_perm_g, ll_q_g, log_rep_g = self.rep_computer(g[idx_g]['G'], params=parameterizations[idx_g],
                                                         device=self.args.device, M=m,
                                                         nobfs=self.args.nobfs, max_cr_iteration=self.args.max_cr_iteration)
            # batch_perms.append(perms)
            log_reps[idx_g].copy_(log_rep_g)
            log_probs[idx_g].copy_(ll_q_g)

            for idx_m in range(m):
                batch_perms[idx_m].append(batch_perm_g[idx_m])

        return batch_perms, log_probs, log_reps


class sequential_net1(nn.Module):

    def __init__(self, args, len_node_vec, hidden_dim=32, vf2=False):
        super().__init__()

        if args.gcn_type == 'gcn':
            self.node_embedding = GCNNet(args, len_node_vec, out_dim=hidden_dim ).to(args.device)
        elif args.gcn_type == 'gat':
            self.node_embedding = GATNet(args, len_node_vec, out_dim=hidden_dim).to(args.device)
        elif args.gcn_type == 'appnp':
            self.node_embedding = APPNET(args, len_node_vec, out_dim=hidden_dim).to(args.device)

        self.node_readout = MLPReadout(hidden_dim, 1).to(args.device)
        self.args = args
        self.rep_computer = mp_sampler(args, vf2)
        self.sample_time = 0

    def forward(self, g, m=1):
        # m: sample_size
        # input embedding
        unbatch_dG = [graph['dG'].to(self.args.device) for graph in g for _ in range(m)]

        batch_dG = dgl.batch(unbatch_dG)
        batch_dGX = batch_dG.ndata['feat'].long()

        batch_num_nodes = batch_dG.batch_num_nodes()

        max_step = max(batch_num_nodes)

        perms = [[] for _ in batch_num_nodes]
        candidate_nodes = [list(range(num_nodes)) for num_nodes in batch_num_nodes]
        remaining_graphs = [i for i in range(len(unbatch_dG)) if batch_num_nodes[i] > len(perms[i])]
        log_probs = torch.zeros(self.args.batch_size * m, device=self.args.device)#, requires_grad=True)

        batch_dG_t = batch_dG
        batch_dGX_t = batch_dGX
        candidate_nodes_t = candidate_nodes
        for t in range(max_step):
            perms_t = [perms[rg] for rg in remaining_graphs]

            node_embedds_t = self.node_embedding(batch_dG_t, batch_dGX_t, perms_t)
            parameterizations = self.node_readout(node_embedds_t).squeeze()

            parameterizations = torch.split(parameterizations, batch_num_nodes[remaining_graphs].tolist())

            nodes_t, log_probs_t = self.sample(parameterizations, candidate_nodes_t)

            log_probs[remaining_graphs] += log_probs_t

            # update permutation up to time step t
            for node_id, rg in zip(nodes_t, remaining_graphs):
                perms[rg].append(node_id)

            # -----------------------------------preparation for step t+1-----------------------------------

            # update remaining_graphs: correct
            remaining_graphs = [i for i in range(self.args.batch_size*m) if batch_num_nodes[i] > len(perms[i])]

            if not remaining_graphs:
                break

            # update candidate_nodes: correct
            candidate_nodes_t = [candidate_nodes[rg] for rg in remaining_graphs]

            # update dgl graph
            batch_dG_t = dgl.batch([unbatch_dG[rg] for rg in remaining_graphs])
            batch_dGX_t = batch_dG_t.ndata['feat'].long()

        log_reps = torch.zeros(self.args.batch_size*m, requires_grad=False, device=self.args.device)

        for i in range(self.args.batch_size*m):
            if self.args.note == "DGMG":
                log_rep = self.rep_computer.compute_repetition(g[i//m]['G'], perms[i])
            else:
                log_rep = compute_autogrp_n(g[i//m]['G'])
            #log_rep = self.rep_computer.compute_repetition(g[i // m]['G'], perms[i])  #use color-refinement for all model
            log_reps[i].fill_(log_rep)

        perms = [perms[i::m] for i in range(m)]

        log_probs = log_probs.view(self.args.batch_size, m)
        log_reps = log_reps.view(self.args.batch_size, m)
        return perms, log_probs, log_reps

    def sample(self, params, candidates):
        nodes = []
        log_probs = torch.zeros(len(candidates), device=self.args.device)
        for b, (param, candidate) in enumerate(zip(params, candidates)):
            categorical_params = F.softmax(param[candidate], dim=0)
            fake_node_id = torch.multinomial(categorical_params, 1).item()
            real_node_id = candidate[fake_node_id]
            nodes.append(real_node_id)
            candidates[b].remove(real_node_id)
            log_probs[b] = torch.log(categorical_params[fake_node_id])

        return nodes, log_probs


class sequential_net2(nn.Module):
    def __init__(self, args, len_node_vec, hidden_dim=32, vf2=False):
        super().__init__()

        if args.gcn_type == 'gcn':
            self.node_embedding = GCNNet(args, 2*len_node_vec, out_dim=hidden_dim).to(args.device)
        elif args.gcn_type == 'gat':
            self.node_embedding = GATNet(args, 2*len_node_vec, out_dim=hidden_dim).to(args.device)
        elif args.gcn_type == 'appnp':
            self.node_embedding = APPNET(args, 2*len_node_vec, out_dim=hidden_dim).to(args.device)
        self.node_readout = MLPReadout(hidden_dim, 1).to(args.device)
        self.args = args
        self.rep_computer = mp_sampler(args, vf2)
        self.len_node_vec = len_node_vec

    def forward(self, g, m=1):
        # m: sample_size
        # input embedding
        unbatch_dG = [graph['dG'].to(self.args.device) for graph in g for _ in range(m)]

        batch_dG = dgl.batch(unbatch_dG)
        batch_dGX = batch_dG.ndata['feat'].long()

        batch_num_nodes = batch_dG.batch_num_nodes()

        max_step = max(batch_num_nodes)

        perms = [[] for _ in batch_num_nodes]
        candidate_nodes = [list(range(num_nodes)) for num_nodes in batch_num_nodes]
        remaining_graphs = [i for i in range(len(unbatch_dG)) if batch_num_nodes[i] > len(perms[i])]
        log_probs = torch.zeros(self.args.batch_size * m, device=self.args.device)  # , requires_grad=True)

        batch_dG_t = batch_dG
        batch_dGX_t = batch_dGX
        candidate_nodes_t = candidate_nodes
        for t in range(max_step):
            perms_t = [perms[rg] for rg in remaining_graphs]

            # st = time.time()
            node_embedds_t = self.node_embedding(batch_dG_t, batch_dGX_t)
            parameterizations = self.node_readout(node_embedds_t).squeeze()
            # self.sample_time += time.time()-st

            # get the node value for each graph: [nodes, nodes, ...]
            parameterizations = torch.split(parameterizations, batch_num_nodes[remaining_graphs].tolist())

            # sample t step node and compute the log prob AND update candidate_nodes for time t+1

            nodes_t, log_probs_t = self.sample(parameterizations, candidate_nodes_t)

            # update log prob
            log_probs[remaining_graphs] += log_probs_t

            # update permutation up to time step t

            ig = 0  #id of graph
            for node_id, rg in zip(nodes_t, remaining_graphs):
                perms[rg].append(node_id)
                #change colors
                unbatch_dG[rg].ndata['feat'][nodes_t[ig], 0] = unbatch_dG[rg].ndata['feat'][nodes_t[ig], 0] + self.len_node_vec
                ig= ig + 1

            # -----------------------------------preparation for step t+1-----------------------------------

                # update remaining_graphs: correct
            remaining_graphs = [i for i in range(self.args.batch_size * m) if batch_num_nodes[i] > len(perms[i])]

            if not remaining_graphs:
                break

            # update candidate_nodes: correct
            candidate_nodes_t = [candidate_nodes[rg] for rg in remaining_graphs]

            # update dgl graph

            batch_dG_t = dgl.batch([unbatch_dG[rg] for rg in remaining_graphs])
            batch_dGX_t = batch_dG_t.ndata['feat'].long()

        log_reps = torch.zeros(self.args.batch_size*m, requires_grad=False, device=self.args.device)

        for i in range(self.args.batch_size * m):

            log_rep = self.rep_computer.compute_repetition(g[i // m]['G'], perms[i])
            log_reps[i].fill_(log_rep)
            # print(self.sample_time)
            # self.sample_time = 0

        perms = [perms[i::m] for i in range(m)]

        log_probs = log_probs.view(self.args.batch_size, m)
        log_reps = log_reps.view(self.args.batch_size, m)

        return perms, log_probs, log_reps

    def sample(self, params, candidates):
        nodes = []
        log_probs = torch.zeros(len(candidates), device=self.args.device)
        for b, (param, candidate) in enumerate(zip(params, candidates)):
            categorical_params = F.softmax(param[candidate], dim=0)
            fake_node_id = torch.multinomial(categorical_params, 1).item()
            real_node_id = candidate[fake_node_id]
            nodes.append(real_node_id)
            candidates[b].remove(real_node_id)
            log_probs[b] = torch.log(categorical_params[fake_node_id])

        return nodes, log_probs


class sequential_net_dfscode(nn.Module):
    def __init__(self, args, len_node_vec, hidden_dim=32):
        super().__init__()

        if args.gcn_type == 'gcn':
            self.node_embedding = GCNNet(args, len_node_vec, out_dim=hidden_dim).to(args.device)
        elif args.gcn_type == 'gat':
            self.node_embedding = GATNet(args, len_node_vec, out_dim=hidden_dim).to(args.device)
        elif args.gcn_type == 'appnp':
            self.node_embedding = APPNET(args, len_node_vec, out_dim=hidden_dim).to(args.device)
        self.node_readout = MLPReadout(hidden_dim, 1).to(args.device)
        self.args = args
        self.rep_computer = mp_sampler(args)
        self.len_node_vec = len_node_vec

    def forward(self, g, m=1):
        # m: sample_size
        # input embedding
        unbatch_dG = [graph['dG'].to(self.args.device) for graph in g for _ in range(m)]
        #net_G
        unbatch_G = [graph['G'] for graph in g for _ in range(m)]

        batch_dG = dgl.batch(unbatch_dG)
        batch_dGX = batch_dG.ndata['feat'].long()

        batch_num_nodes = batch_dG.batch_num_nodes()

        max_step = max(batch_num_nodes)
        #relabel node for dfs_code ^-^
        node_map = [{} for g in unbatch_dG]
        re_node_map = [{} for g in unbatch_dG]
        #record added node
        added_nodes = [set() for g in unbatch_dG]
        #dfs_code
        dfs_codes = [[] for g in unbatch_dG]



        perms = [[] for _ in batch_num_nodes]
        candidate_nodes = [list(range(num_nodes)) for num_nodes in batch_num_nodes]
        remaining_graphs = [i for i in range(len(unbatch_dG)) if batch_num_nodes[i] > len(perms[i])]
        log_probs = torch.zeros(self.args.batch_size * m, device=self.args.device)  # , requires_grad=True)

        batch_dG_t = batch_dG
        batch_dGX_t = batch_dGX
        unbatch_G_t = unbatch_G
        added_nodes_t = added_nodes
        candidate_nodes_t = candidate_nodes
        dfs_codes_t = dfs_codes
        re_node_map_t = re_node_map
        node_map_t = node_map
        for t in range(max_step):
            perms_t = [perms[rg] for rg in remaining_graphs]

            # st = time.time()
            node_embedds_t = self.node_embedding(batch_dG_t, batch_dGX_t, perms_t)
            parameterizations = self.node_readout(node_embedds_t).squeeze()
            # self.sample_time += time.time()-st

            # get the node value for each graph: [nodes, nodes, ...]
            parameterizations = torch.split(parameterizations, batch_num_nodes[remaining_graphs].tolist())

            # sample t step node and compute the log prob AND update candidate_nodes for time t+1

            nodes_t, log_probs_t = self.sample(parameterizations, candidate_nodes_t, unbatch_G_t, added_nodes_t, t)


            #update map
            for index, map in enumerate(node_map_t):
                map[nodes_t[index]] = t
                re_node_map_t[index][t] = nodes_t[index]
                # update candidate_nodes
                candidate_nodes[remaining_graphs[index]]=candidate_nodes_t[index]
            #TODO: import implementation: create dfs_code

            for i in range(len(nodes_t)):
                #current_graph
                graph_cur = unbatch_G_t[i]
                # find edges to be added to code
                neighbors = [node for node in graph_cur.neighbors(nodes_t[i]) if node in added_nodes_t[i]]
                #map to code number space, and sort
                node_map_acc = node_map_t[i]
                code_ns = [node_map_acc[node] for node in neighbors]
                code_ns.sort()
                #new node map
                cur_node = node_map_acc[nodes_t[i]]
                #append edges to dfs_codes
                for node in code_ns:
                    #always from old to new
                    new_code = [node, cur_node, graph_cur.nodes[re_node_map_t[i][node]]['label'], graph_cur.edges[re_node_map_t[i][node],
                                 nodes_t[i]]['label'], graph_cur.nodes[nodes_t[i]]['label']]
                    dfs_codes_t[i].append(new_code)

            # update log prob
            log_probs[remaining_graphs] += log_probs_t

            # update permutation up to time step t

            ig = 0  #id of graph
            for node_id, rg in zip(nodes_t, remaining_graphs):
                perms[rg].append(node_id)
                #change colors
                #unbatch_dG[rg].ndata['feat'][nodes_t[ig], 0] = unbatch_dG[rg].ndata['feat'][nodes_t[ig], 0] + self.len_node_vec
                ig= ig + 1

            # -----------------------------------preparation for step t+1-----------------------------------

                # update remaining_graphs: correct
            remaining_graphs = [i for i in range(self.args.batch_size * m) if batch_num_nodes[i] > len(perms[i])]

            if not remaining_graphs:
                break

            # update candidate_nodes: correct--interesting for .remove()

            candidate_nodes_t = [candidate_nodes[rg] for rg in remaining_graphs]

            # update dgl graph

            batch_dG_t = dgl.batch([unbatch_dG[rg] for rg in remaining_graphs])
            batch_dGX_t = batch_dG_t.ndata['feat'].long()
            unbatch_G_t = [unbatch_G[rg] for rg in remaining_graphs]
            added_nodes_t = [added_nodes[rg] for rg in remaining_graphs]
            dfs_codes_t = [dfs_codes[rg] for rg in remaining_graphs]
            re_node_map_t = [re_node_map[rg] for rg in remaining_graphs]
            node_map_t = [node_map[rg] for rg in remaining_graphs]

        log_reps = torch.zeros(self.args.batch_size*m, requires_grad=False, device=self.args.device)

        for i in range(self.args.batch_size * m):


            log_rep = self.rep_computer.compute_repetition(g[i//m]['G'], perms[i])

            # print(self.sample_time)
            # self.sample_time = 0

        perms = [perms[i::m] for i in range(m)]

        log_probs = log_probs.view(self.args.batch_size, m)
        log_reps = log_reps.view(self.args.batch_size, m)


        #finsh dfs_codes in this function
        return perms, log_probs, log_reps, dfs_codes


    def sample(self, params, candidates, unbatch_G, added_nodes, iter):
        nodes = []
        log_probs = torch.zeros(len(candidates), device=self.args.device)
        for b, (param, candidate) in enumerate(zip(params, candidates)):
            categorical_params = F.softmax(param[candidate], dim=0)
            fake_node_id = torch.multinomial(categorical_params, 1).item()
            real_node_id = candidate[fake_node_id]
            added_nodes[b].add(real_node_id)
            nodes.append(real_node_id)
            #candidates[b].remove(real_node_id)
            log_probs[b] = torch.log(categorical_params[fake_node_id])
            #reviced directly on list, will influence global list
            if iter != 0:
                #candidates[b] = list(set(candidates[b]).update([n for n in unbatch_G[b].neighbors(real_node_id)]).difference_update(added_nodes[b]))
                candidates[b] = set(candidates[b])
                candidates[b].update([n for n in unbatch_G[b].neighbors(real_node_id)])


                candidates[b].difference_update(added_nodes[b])
                candidates[b] = list(candidates[b])
            elif iter == 0:
                candidates[b] = [n for n in unbatch_G[b].neighbors(real_node_id)]

        return nodes, log_probs

# class sequential_net4(nn.Module):
#     def __init__(self, args, len_node_vec, hidden_dim=32):
#         super().__init__()
#
#
#         self.node_embedding = APPNET(args, 2*len_node_vec, out_dim=hidden_dim).to(args.device)
#         self.node_readout = MLPReadout(hidden_dim, 1).to(args.device)
#         self.args = args
#         self.rep_computer = mp_sampler(args)
#         self.len_node_vec = len_node_vec
#
#     def forward(self, g, m=1):
#         # m: sample_size
#         # input embedding
#         unbatch_dG = [graph['dG'].to(self.args.device) for graph in g for _ in range(m)]
#
#         batch_dG = dgl.batch(unbatch_dG)
#         batch_dGX = batch_dG.ndata['feat'].long()
#
#         batch_num_nodes = batch_dG.batch_num_nodes()
#
#         max_step = max(batch_num_nodes)
#
#         perms = [[] for _ in batch_num_nodes]
#         candidate_nodes = [list(range(num_nodes)) for num_nodes in batch_num_nodes]
#         remaining_graphs = [i for i in range(len(unbatch_dG)) if batch_num_nodes[i] > len(perms[i])]
#         log_probs = torch.zeros(self.args.batch_size * m, device=self.args.device)  # , requires_grad=True)
#
#         batch_dG_t = batch_dG
#         batch_dGX_t = batch_dGX
#         candidate_nodes_t = candidate_nodes
#
#         cur_n_graphs = len(remaining_graphs)
#         for t in range(max_step):
#             perms_t = [perms[rg] for rg in remaining_graphs]
#
#             node_embedds_t = self.node_embedding(batch_dG_t, batch_dGX_t, perms_t)
#             parameterizations = self.node_readout(node_embedds_t).squeeze()
#
#             # get the node value for each graph: [nodes, nodes, ...]
#             parameterizations = torch.split(parameterizations, batch_num_nodes[remaining_graphs].tolist())
#
#             # sample t step node and compute the log prob AND update candidate_nodes for time t+1
#
#             nodes_t, log_probs_t = self.sample(parameterizations, candidate_nodes_t)
#
#             # update log prob
#             log_probs[remaining_graphs] += log_probs_t
#
#             # update permutation up to time step t
#             ig = 0
#             for node_id, rg in zip(nodes_t, remaining_graphs):
#                 # unbatch_dG[rg].ndata['feat'][nodes_t[ig], 0] = unbatch_dG[rg].ndata['feat'][nodes_t[ig], 0] + self.len_node_vec
#                 perms[rg].append(node_id)
#                 ig = ig + 1
#
#             # -----------------------------------preparation for step t+1-----------------------------------
#
#                 # update remaining_graphs:
#                 remaining_graphs = [i for i in range(self.args.batch_size * m) if batch_num_nodes[i] > len(perms[i])]
#
#                 if not remaining_graphs:
#                     break
#
#                 # update candidate_nodes:
#                 candidate_nodes_t = [candidate_nodes[rg] for rg in remaining_graphs]
#                 # update dgl graph
#                 if cur_n_graphs != len(remaining_graphs):
#                     batch_dG_t = dgl.batch([unbatch_dG[rg] for rg in remaining_graphs])
#                     batch_dGX_t = batch_dG_t.ndata['feat'].long()
#                     cur_n_graphs = len(remaining_graphs)
#                     # adj = batch_dG_t.adjacency_matrix(scipy_fmt="csr")
#                     # self.node_embedding.setup_propagator(adj)
#
#         log_reps = torch.zeros(self.args.batch_size*m, requires_grad=False, device=self.args.device)
#
#         for i in range(self.args.batch_size * m):
#             log_rep = self.rep_computer.compute_repetition(g[i // m]['G'], perms[i])
#             log_reps[i].fill_(log_rep)
#
#         perms = [perms[i::m] for i in range(m)]
#
#         log_probs = log_probs.view(self.args.batch_size, m)
#         log_reps = log_reps.view(self.args.batch_size, m)
#
#         return perms, log_probs, log_reps
#
#     def sample(self, params, candidates):
#         nodes = []
#         log_probs = torch.zeros(len(candidates), device=self.args.device)
#         for b, (param, candidate) in enumerate(zip(params, candidates)):
#             categorical_params = F.softmax(param[candidate], dim=0)
#             fake_node_id = torch.multinomial(categorical_params, 1).item()
#             real_node_id = candidate[fake_node_id]
#             nodes.append(real_node_id)
#             candidates[b].remove(real_node_id)
#             log_probs[b] = torch.log(categorical_params[fake_node_id])
#
#         return nodes, log_probs
