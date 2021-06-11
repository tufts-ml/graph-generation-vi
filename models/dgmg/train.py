"""
Code adapted from https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgmg
"""

import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F
from models.dgmg.model import message_passing, calc_graph_embedding, calc_init_embedding, DGM_graphs
from torch.autograd import Variable
from utils import load_model, get_model_attribute




def evaluate_loss(model, data):
    model.zero_grad()
    is_fast = False
    # graph = dataset[i]
    # # do random ordering: relabel nodes
    # node_order = list(range(graph.number_of_nodes()))
    # shuffle(node_order)
    # order_mapping = dict(zip(graph.nodes(), node_order))
    # graph = nx.relabel_nodes(graph, order_mapping, copy=True)

    # NOTE: when starting loop, we assume a node has already been generated
    graph = data[0]
    node_count = 1
    node_embedding = [
        Variable(torch.ones(1, model.h_size)).cuda()]  # list of torch tensors, each size: 1*hidden

    loss = 0
    # print(graph.nodes())
    while node_count <= graph.number_of_nodes():
        sg1 = graph.subgraph(list(range(node_count)))
        node_neighbor = [list(sg1.neighbors(i)) for i in range(node_count)]

        # 1 message passing
        # do 2 times message passing
        node_embedding = message_passing(node_neighbor, node_embedding, model)

        # 2 graph embedding and new node embedding
        node_embedding_cat = torch.cat(node_embedding, dim=0)
        graph_embedding = calc_graph_embedding(node_embedding_cat, model)
        init_embedding = calc_init_embedding(node_embedding_cat, model)

        # 3 f_addnode
        p_addnode = model.f_an(graph_embedding)
        if node_count < graph.number_of_nodes():
            # add node
            node_neighbor.append([])
            node_embedding.append(init_embedding)
            if is_fast:
                node_embedding_cat = torch.cat(node_embedding, dim=0)
            # calc loss
            loss_addnode_step = F.binary_cross_entropy(p_addnode, Variable(torch.ones((1, 1))).cuda())
            # loss_addnode_step.backward(retain_graph=True)
            loss += loss_addnode_step
            # loss_addnode += loss_addnode_step.data
        else:
            # calc loss
            loss_addnode_step = F.binary_cross_entropy(p_addnode, Variable(torch.zeros((1, 1))).cuda())
            # loss_addnode_step.backward(retain_graph=True)
            loss += loss_addnode_step
            # loss_addnode += loss_addnode_step.data
            break

        sg2 = graph.subgraph(list(range(node_count+1)))
        # print('node_neighbor', node_neighbor)
        node_neighbor_new = list(sg2.neighbors(node_count))
        edge_count = 0
        while edge_count <= len(node_neighbor_new): # add node_neighbor (
            if not is_fast:
                node_embedding = message_passing(node_neighbor, node_embedding, model)
                node_embedding_cat = torch.cat(node_embedding, dim=0)
                graph_embedding = calc_graph_embedding(node_embedding_cat, model)

            # 4 f_addedge
            p_addedge = model.f_ae(graph_embedding)

            if edge_count < len(node_neighbor_new):
                # calc loss
                loss_addedge_step = F.binary_cross_entropy(p_addedge, Variable(torch.ones((1, 1))).cuda())
                # loss_addedge_step.backward(retain_graph=True)
                loss += loss_addedge_step
                # loss_addedge += loss_addedge_step.data

                # 5 f_nodes
                # excluding the last node (which is the new node)
                node_new_embedding_cat = node_embedding_cat[-1, :].expand(node_embedding_cat.size(0) - 1,
                                                                          node_embedding_cat.size(1))
                s_node = model.f_s(torch.cat((node_embedding_cat[0:-1, :], node_new_embedding_cat), dim=1))
                p_node = F.softmax(s_node.permute(1, 0))
                # get ground truth
                a_node = torch.zeros((1, p_node.size(1)))
                # print('node_neighbor_new',node_neighbor_new, edge_count)
                a_node[0, node_neighbor_new[edge_count]] = 1
                a_node = Variable(a_node).cuda()
                # add edge
                node_neighbor[-1].append(node_neighbor_new[edge_count])
                node_neighbor[node_neighbor_new[edge_count]].append(len(node_neighbor) - 1)
                # calc loss
                loss_node_step = F.binary_cross_entropy(p_node, a_node, reduction="sum")
                # loss_node_step.backward(retain_graph=True)
                loss += loss_node_step
                # loss_node += loss_node_step.data

            else:
                # calc loss
                loss_addedge_step = F.binary_cross_entropy(p_addedge, Variable(torch.zeros((1, 1))).cuda())
                # loss_addedge_step.backward(retain_graph=True)
                loss += loss_addedge_step
                # loss_addedge += loss_addedge_step.data
                break

            edge_count += 1
        node_count += 1
    return loss


def gumbel_softmax(logits, temperature, eps=1e-9):
    '''
    :param logits: shape: N*L
    :param temperature:
    :param eps:
    :return:
    '''
    # get gumbel noise
    noise = torch.rand(logits.size())
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    noise = Variable(noise).cuda()

    x = (logits + noise) / temperature
    x = F.softmax(x)
    return x


def sample_tensor(y,sample=True, thresh=0.5):
    # do sampling
    if sample:
        y_thresh = Variable(torch.rand(y.size())).cuda()
        y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size())*thresh).cuda()
        y_result = torch.gt(y, y_thresh).float()
    return y_result

def predict_graphs(eval_args, is_fast=False):
    args = eval_args.train_args
    model = {'dgmg':DGM_graphs(args.feat_size).cuda()}
    load_model(eval_args.model_path, eval_args.device, model)
    model = model['dgmg']
    model.eval()
    graph_num = eval_args.count

    graphs_generated = []
    i=0
    while i < graph_num:
        # NOTE: when starting loop, we assume a node has already been generated
        node_neighbor = [[]]  # list of lists (first node is zero)
        node_embedding = [Variable(torch.ones(1,args.feat_size)).cuda()] # list of torch tensors, each size: 1*hidden

        node_count = 1
        while node_count<=args.max_prev_node:
            # 1 message passing
            # do 2 times message passing
            node_embedding = message_passing(node_neighbor, node_embedding, model)

            # 2 graph embedding and new node embedding
            node_embedding_cat = torch.cat(node_embedding, dim=0)
            graph_embedding = calc_graph_embedding(node_embedding_cat, model)
            init_embedding = calc_init_embedding(node_embedding_cat, model)

            # 3 f_addnode
            p_addnode = model.f_an(graph_embedding)
            a_addnode = sample_tensor(p_addnode)
            # print(a_addnode.data[0][0])
            if a_addnode.data[0][0]==1:
                # print('add node')
                # add node
                node_neighbor.append([])
                node_embedding.append(init_embedding)
                if is_fast:
                    node_embedding_cat = torch.cat(node_embedding, dim=0)
            else:
                break

            edge_count = 0
            while edge_count<args.max_prev_node:
                if not is_fast:
                    node_embedding = message_passing(node_neighbor, node_embedding, model)
                    node_embedding_cat = torch.cat(node_embedding, dim=0)
                    graph_embedding = calc_graph_embedding(node_embedding_cat, model)

                # 4 f_addedge
                p_addedge = model.f_ae(graph_embedding)
                a_addedge = sample_tensor(p_addedge)
                # print(a_addedge.data[0][0])

                if a_addedge.data[0][0]==1:
                    # print('add edge')
                    # 5 f_nodes
                    # excluding the last node (which is the new node)
                    node_new_embedding_cat = node_embedding_cat[-1,:].expand(node_embedding_cat.size(0)-1,node_embedding_cat.size(1))
                    s_node = model.f_s(torch.cat((node_embedding_cat[0:-1,:],node_new_embedding_cat),dim=1))
                    p_node = F.softmax(s_node.permute(1,0))
                    a_node = gumbel_softmax(p_node, temperature=0.01)
                    _, a_node_id = a_node.topk(1)
                    a_node_id = int(a_node_id.data[0][0])
                    # add edge
                    node_neighbor[-1].append(a_node_id)
                    node_neighbor[a_node_id].append(len(node_neighbor)-1)
                else:
                    break

                edge_count += 1
            node_count += 1
        if node_count>=8:
            i+=1
            # save graph
            node_neighbor_dict = dict(zip(list(range(len(node_neighbor))), node_neighbor))
            graph = nx.from_dict_of_lists(node_neighbor_dict)
            graphs_generated.append(graph)

    graphs = []
    nb = "DEFAULT_LABEL"
    eb = "DEFAULT_LABEL"
    for graph in graphs_generated:
        labeled_graph = nx.Graph()

        for v in graph.nodes():
            labeled_graph.add_node(
                v, label=nb)

        for u, v in graph.edges():
            labeled_graph.add_edge(
                u, v, label=eb)

        graphs.append(labeled_graph)

    return graphs
