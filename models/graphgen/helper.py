import torch
import networkx as nx


def dfscode_to_tensor(dfscode, feature_map):
    max_nodes, max_edges = feature_map['max_nodes'], feature_map['max_edges']
    node_forward_dict, edge_forward_dict = feature_map['node_forward'], feature_map['edge_forward']
    num_nodes_feat, num_edges_feat = len(
        feature_map['node_forward']), len(feature_map['edge_forward'])

    # max_nodes, num_nodes_feat and num_edges_feat are end token labels
    # So ignore tokens are one higher
    dfscode_tensors = {
        't1': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        't2': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'v1': (num_nodes_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'e': (num_edges_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'v2': (num_nodes_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'len': len(dfscode)
    }

    for i, code in enumerate(dfscode):
        dfscode_tensors['t1'][i] = int(code[0])
        dfscode_tensors['t2'][i] = int(code[1])
        dfscode_tensors['v1'][i] = int(node_forward_dict[code[2]])

        dfscode_tensors['e'][i] = int(edge_forward_dict[code[3]])
        dfscode_tensors['v2'][i] = int(node_forward_dict[code[4]])

    # Add end token
    dfscode_tensors['t1'][len(dfscode)], dfscode_tensors['t2'][len(
        dfscode)] = max_nodes, max_nodes
    dfscode_tensors['v1'][len(dfscode)], dfscode_tensors['v2'][len(
        dfscode)] = num_nodes_feat, num_nodes_feat
    dfscode_tensors['e'][len(dfscode)] = num_edges_feat

    return dfscode_tensors

def graph_from_dfscode(dfscode):
    graph = nx.Graph()

    for dfscode_egde in dfscode:
        i, j, l1, e, l2 = dfscode_egde
        graph.add_node(int(i), label=l1)
        graph.add_node(int(j), label=l2)
        graph.add_edge(int(i), int(j), label=e)

    return graph