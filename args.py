from datetime import datetime
import torch
from utils import get_model_attribute
import argparse
import os

class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # logging & saving
        self.parser.add_argument('--clean_tensorboard', action='store_true', help='Clean tensorboard')
        self.parser.add_argument('--clean_temp', action='store_true', help='Clean temp folder')
        self.parser.add_argument('--log_tensorboard', action='store_true', help='Whether to use tensorboard for logging')
        self.parser.add_argument('--save_model', default=True, action='store_true', help='Whether to save model')
        self.parser.add_argument('--print_interval', type=int, default=1, help='loss printing batch interval')
        self.parser.add_argument('--epochs_save', type=int, default=10, help='model save epoch interval')
        self.parser.add_argument('--epochs_validate', type=int, default=1, help='model validate epoch interval')

        # setup
        self.parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='cuda:[d] | cpu')
        self.parser.add_argument('--use_baseline', default=False ,action='store_true', help='Whether to run baseline or proposed method')
        self.parser.add_argument('--enable_gcn', default=True, action='store_true', help='Whether to q or uniform distribution for sampling')

        self.parser.add_argument('--note', default='GraphRNN', help='Algorithm Version -- GraphRNN | DGMG (Deep GMG)| Graphgen| GRAN')


        self.parser.add_argument('--seed', type=int, default=123, help='random seed to reproduce performance/dataset')

        # dataset

        self.parser.add_argument('--graph_type', default='caveman_small', help='Select dataset to train the model')
        self.parser.add_argument('--num_graphs', type=int, default=None, help='take complete dataset(None) | part of dataset')
        self.parser.add_argument('--produce_graphs', default=True, action='store_true', help='Whether to reproduce graphs')
        self.parser.add_argument('--label', default=False, action='store_true', help='Whether to use label infomation in dataset')
        self.parser.add_argument('--data_small', default=False, action='store_true',
                                 help='Whether to use only 300 samples to train')


        # Specific to sampler
        self.parser.add_argument('--sample_size', type=int, default=4, help='sample size for gradient estimator')
        self.parser.add_argument('--max_cr_iteration', type=int, default=3, help='maximum color refinement for computing multiplicity')
        self.parser.add_argument('--use_mp_sampler',  default=True, action='store_true', help='Whether to multi-process for permutation sampler')
        self.parser.add_argument('--mp_num_workers', type=int, default=4, help='number of workers for permutation sampling')
        self.parser.add_argument('--sample_arch', default='position', help='type of sequential sampling style: { relabel|postion|ORI }')

        # Specific to GraphRNN
        self.parser.add_argument('--max_prev_node', type=int, default=None, help='max previous node that looks back for GraphRNN')
        self.parser.add_argument('--hidden_size_node_level_rnn', type=int, default=128, help='hidden size for node level RNN')
        self.parser.add_argument('--embedding_size_node_level_rnn', type=int, default=64, help='the size for node level RNN input')
        self.parser.add_argument('--embedding_size_node_output', type=int, default=64, help='the size of node output embedding')
        self.parser.add_argument('--hidden_size_edge_level_rnn', type=int, default=16, help='hidden size for edge level RNN')
        self.parser.add_argument('--embedding_size_edge_level_rnn', type=int, default=8, help='the size for edge level RNN input')
        self.parser.add_argument('--embedding_size_edge_output', type=int, default=8, help='the size of edge output embedding')
        self.parser.add_argument('--num_layers', type=int, default=4, help='layers of rnn')
        self.parser.add_argument('--nobfs', default=True, action='store_true', help='whether to use bfs constraint during sampling')

        # Specific to DGMG
        self.parser.add_argument('--feat_size', type=int, default=128, help='feature size')
        self.parser.add_argument('--hops', type=int, default=2, help='')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='')

        # Specific to Graphgen

        self.parser.add_argument('--embedding_size_dfscode_rnn', type=int, default=92, help='input size for dfscode RNN')
        self.parser.add_argument('--embedding_size_timestamp_output', type=int, default=512)
        self.parser.add_argument('--embedding_size_vertex_output', type=int, default=512,
                                 help='the size for vertex output embedding')
        self.parser.add_argument('--dfscode_embedding_size_edge_output', type=int, default=512,
                                 help='ithe size for edge output embedding')
        self.parser.add_argument('--dfscode_rnn_dropout', type=float, default=0.2,
                                 help='Dropout layer in between RNN layers')
        # Specific to GRAN
        self.parser.add_argument('--num_mix_component', type=int, default=20, help='a')
        self.parser.add_argument('--is_sym', default=True, help='a')
        self.parser.add_argument('--block_size', type=int, default=1, help='1/2')
        self.parser.add_argument('--sample_stride', type=int, default=1, help='a')
        self.parser.add_argument('--hidden_dim', type=int, default=128, help='a')
        self.parser.add_argument('--embedding_dim', type=int, default=128, help='a')
        self.parser.add_argument('--num_GNN_layers', type=int, default=7, help='a')
        self.parser.add_argument('--num_GNN_prop', type=int, default=1, help='a')
        self.parser.add_argument('--dimension_reduce', default=True, help='a')
        self.parser.add_argument('--has_attention', default=True, help='a')
        self.parser.add_argument('--num_canonical_order', type=int, default=1, help="must be 1 during training")
        self.parser.add_argument('--edge_weight', type=float, default=1.0e+0, help="only used for baseline")
        self.parser.add_argument('--num_fwd_pass', type=int, default=1, help="1")


        # training config
        self.parser.add_argument('--batch_size', type=int, default=1, help='batchsize')
        self.parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataloader')
        self.parser.add_argument('--epochs', type=int, default=5000, help='epochs')

        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.3, help='Learning rate decay factor')

        # Model load parameters
        self.parser.add_argument('--load_model',  default=False, action='store_true', help='whether to load model')
        self.parser.add_argument('--load_model_path', default='output/GRAN_Lung_unif_nobfs_2021_01_24_23_55_12/', help='load model path')
        self.parser.add_argument('--load_device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='load device: cuda:[d] | cpu')
        self.parser.add_argument('--epochs_end', type=int, default=100, help='model in which epoch to load')



        # Specify to GNN
        self.parser.add_argument('--gcn_type', default='gat', help='type of GNN for q model: { gat | gcn | appnp}')
        self.parser.add_argument('--gcn_hidden_dim', type=int, default=32, help='gnn hidden dimension')
        self.parser.add_argument('--gcn_out_dim', type=int, default=32, help='gnn output dimension')
        self.parser.add_argument('--gcn_in_feat_dropout', type=float, default=0.0, help='gnn input feature dropout rate')
        self.parser.add_argument('--gcn_dropout', type=float, default=0.2, help='gnn hidden feature dropout rate')
        self.parser.add_argument('--gcn_num_layers', type=int, default=3, help='number of layers of gnn')
        self.parser.add_argument('--gcn_batch_norm',  default=True, action='store_true', help='whether to use batchnorm in gnn')
        self.parser.add_argument('--gcn_residual',  default=True, action='store_true', help='whether to residual connection in gnn')
        self.parser.add_argument('--gcn_n_classes',  type=int, default=1, help='')
        self.parser.add_argument('--gat_nheads',  type=int, default=6, help='only applied for GAT model')
        self.parser.add_argument('--gcn_pretrain_path', default='', help='petrained gcn path')

    def update_args(self):
        args = self.parser.parse_args()
        if args.load_model:
            old_args = args
            fname = os.path.join(args.load_model_path, "model_save", "epoch_{}.dat".format(old_args.epochs_end))
            args = get_model_attribute(
                'saved_args', fname, args.load_device)
            args.device = old_args.load_device
            args.load_model = True
            args.load_model_path = old_args.load_model_path
            args.epochs = old_args.epochs
            args.epochs_end = old_args.epochs_end

            args.clean_tensorboard = False
            args.clean_temp = False
            args.produce_graphs = False

            return args


        # some conflict configuration for baseline and propose method should be update here
        if args.use_baseline:
            args.enable_gcn = False
        else:
            if args.enable_gcn:
                pass
            else:
                args.sample_size = 1

        args.milestones = [args.epochs//5, args.epochs//5*2, args.epochs//5*3, args.epochs//5*4]  # List of milestones

        args.time = '{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        if args.use_baseline:
            type = "Base"
        else:
            if args.enable_gcn:
                type = args.gcn_type
            else:
                type = "unif"

        if args.note == 'DGMG':  #nobfs for dgmg
            args.nobfs = True

        bfs = "nobfs" if args.nobfs else "bfs"
        args.fname = args.note + '_' + args.graph_type + "_" + type + "_" + bfs + "_" + args.time
        args.dir_input = 'output/'
        args.experiment_path = args.dir_input + args.fname
        args.logging_path = args.experiment_path + '/' + 'logging/'
        args.logging_iter_path = args.logging_path + 'iter_history.csv'
        args.logging_epoch_path = args.logging_path + 'epoch_history.csv'
        args.model_save_path = args.experiment_path + '/' + 'model_save/'
        args.tensorboard_path = args.experiment_path + '/' + 'tensorboard/'
        args.dataset_path = 'datasets/'
        args.temp_path = args.experiment_path + '/' + 'tmp/'

        args.current_model_save_path = args.model_save_path

        args.load_model_path = None

        args.current_dataset_path = None
        args.current_processed_dataset_path = None
        args.current_temp_path = args.temp_path

        # noise argument for the barabasi_noise graph
        args.noise = 1
        return args
