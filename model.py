# from models.dgmg.model import create_model as create_model_dgmg
from models.dgmg.model import DGM_graphs
from models.graph_rnn.model import create_model as create_model_graph_rnn
from models.gcn.model import create_model as create_model_gcn
from models.graphgen.model import create_model as create_model_graphgen
# from models.gran.model import create_model as create_model_gran
from utils import load_model, get_last_checkpoint


def create_models(args, feature_map, vf2=False):
    print('Producing model...')
    if args.note == 'GraphRNN':
        model = create_model_graph_rnn(args, feature_map)

    elif args.note == 'DGMG':
        model = {'dgmg': DGM_graphs(args.feat_size).cuda()}#create_model_dgmg(args, feature_map)

    elif args.note == 'Graphgen':
        model = create_model_graphgen(args, feature_map)
    elif args.note == 'GRAN':
        model = create_model_gran(args, feature_map)

    gcn = create_model_gcn(args, feature_map, vf2)
    return model, gcn
