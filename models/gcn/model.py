from models.gcn.net.gcn_net_node import GCNNet
from models.gcn.net.gat_net_node import GATNet
from models.gcn.SqNet import *


def create_model(args, feature_map, vf2=False):
    #used for embedding layer in GCN model
    len_node_vec = len(feature_map['node_forward'])

    if args.sample_arch == "ORI":
        model = sequential_net0(args, len_node_vec, vf2)
    if args.sample_arch == "position":
        model = sequential_net1(args, len_node_vec, args.gcn_out_dim, vf2)
    if args.sample_arch == "relabel":
        model = sequential_net2(args, len_node_vec, args.gcn_out_dim, vf2)
    if args.note == 'Graphgen':
        model = sequential_net_dfscode(args, len_node_vec, args.gcn_out_dim)

    return model