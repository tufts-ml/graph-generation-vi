import random
import time
import pickle
from torch.utils.data import DataLoader
import torch
import os, json
from args import Args
from utils import create_dirs
from datasets.process_dataset import create_graphs
from datasets.preprocess import calc_max_prev_node
from models.data import Graph_from_file
from models.graph_rnn.data import Graph_to_Adj_Matrix, Graph_Adj_Matrix
from models.dgmg.data import Graph_to_Action
from models.gcn.helper import legal_perms_sampler, mp_sampler
from models.graphgen.helper import dfscode_to_tensor
from models.gran.data import GRANData

from model import create_models
from train import train



if __name__ == '__main__':
    args = Args()
    args = args.update_args()

    create_dirs(args)

    random.seed(args.seed)

    # graphs = create_graphs(args)[:100] # for implementation test
    graphs = create_graphs(args)

    random.shuffle(graphs)
    graphs_train = graphs[: int(0.80 * len(graphs))]
    graphs_validate = graphs[int(0.80 * len(graphs)): int(0.90 * len(graphs))]



    # show graphs statistics
    print('Model:', args.note)
    print('Device:', args.device)
    print('Graph type:', args.graph_type)
    print('Training set: {}, Validation set: {}'.format(
        len(graphs_train), len(graphs_validate)))

    # Loading the feature map
    with open(args.current_dataset_path + 'map.dict', 'rb') as f:
        feature_map = pickle.load(f)

    print('Max number of nodes: {}'.format(feature_map['max_nodes']))
    print('Max number of edges: {}'.format(feature_map['max_edges']))
    print('Min number of nodes: {}'.format(feature_map['min_nodes']))
    print('Min number of edges: {}'.format(feature_map['min_edges']))
    print('Max degree of a node: {}'.format(feature_map['max_degree']))
    print('No. of node labels: {}'.format(len(feature_map['node_forward'])))
    print('No. of edge labels: {}'.format(len(feature_map['edge_forward'])))
    print(args.__dict__)

    if not args.use_baseline:
        if args.note == 'GraphRNN':
            start = time.time()
            if args.nobfs:
                args.max_prev_node = feature_map['max_nodes'] - 1
            if args.max_prev_node is None:
                    args.max_prev_node = calc_max_prev_node(args.current_processed_dataset_path)

            args.max_head_and_tail = None
            print('max_prev_node:', args.max_prev_node)

            end = time.time()
            print('Time taken to calculate max_prev_node = {:.3f}s'.format(
                end - start))

        dataset_train = Graph_from_file(args, graphs_train, feature_map)
        dataset_validate = Graph_from_file(args, graphs_validate, feature_map)
        dataloader_train = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.num_workers, collate_fn=dataset_train.collate_batch)
        dataloader_validate = DataLoader(
            dataset_validate, batch_size=args.batch_size, shuffle=False,drop_last=True,
            num_workers=args.num_workers, collate_fn=dataset_validate.collate_batch)
    
        if args.note == 'GraphRNN':
            processor = Graph_to_Adj_Matrix(args, feature_map, random_bfs=True)
        elif args.note == 'DGMG':
            processor = Graph_to_Action(args, feature_map)
        elif args.note == 'Graphgen'  :
            processor = dfscode_to_tensor
        elif args.note == 'GRAN'  :
            processor = GRANData(args,feature_map['max_nodes'])


        if args.use_mp_sampler:
            sample_perm = mp_sampler(args)
        else:
            sample_perm = legal_perms_sampler

        # save args
        with open(os.path.join(args.experiment_path, "configuration.txt"), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        model, gcn = create_models(args, feature_map)
        if args.gcn_pretrain_path:
            gcn_weight = torch.load(args.gcn_pretrain_path)
            gcn.load_state_dict(gcn_weight)
            print("pretrained gcn loaded: {}".format(args.gcn_pretrain_path))
        #
        # gcn_weight = torch.load("/home/golf/Downloads/GraphRNN_Lung_gat_nobfs_2020_12_18_21_13_40/model_save/epoch_1.dat")
        # gcn.load_state_dict(gcn_weight['gcn']['gcn'])
        # for name, param in gcn.named_parameters():
        #     print(name, param)
        train(args, model, gcn, feature_map, dataloader_train, dataloader_validate, processor, sample_perm)

    else:
        train_b(args, graphs_train, graphs_validate, feature_map)





