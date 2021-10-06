import torch
import dgl
import time
import os
import pandas as pd
from collections import defaultdict
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
from utils import save_model, load_model, get_model_attribute, get_last_checkpoint
from models.gcn.helper import legal_perms_sampler
from models.graph_rnn.train import evaluate_loss as eval_loss_graph_rnn
from models.dgmg.train import evaluate_loss as eval_loss_dgmg
# from models.gran.model import evaluate_loss as eval_loss_gran
from models.graphgen.train import evaluate_loss as eval_loss_graphgen
from torch.utils.data._utils.collate import default_collate as collate


def evaluate_loss(args, model, gcn, processor, sample_perm, graphs, feature_map, i_epoch):
    '''
    :param args:
    :param model:
    :param gcn:
    :param processor:
    :param graphs: [{‘G’：networkx, 'dG':dglgraph}]
    :param feature_map:
    :return:
    '''
    # get the batch of dGs
    batch_G = [graph['G'] for graph in graphs]
    real_batch_size = len(batch_G)
    # gcn feed-forward
    #get number of node labels
    len_node_vec = len(feature_map['node_forward'])

    #  ------ update Dec 4: sequential decision ------
    if args.enable_gcn:
        # use gcn for sampling
        # for idx_m in range(args.sample_size):
        if args.note != 'Graphgen':
            batch_perms, ll_q, log_repetitions = gcn(graphs, args.sample_size)
        elif args.note == 'Graphgen':
            #dfs_code_list used for p(g)
            batch_perms, ll_q, log_repetitions, dfs_code_list = gcn(graphs, args.sample_size)




        # log_repetitions[:, idx_m].copy_(log_repetition_m)
        # ll_q[:, idx_m].copy_(ll_q_m)
        # batch_perms.append(batch_perm_m)
    else:
        log_repetitions = torch.empty((real_batch_size, args.sample_size), requires_grad=False) # (N, M)
        ll_q = torch.empty((real_batch_size, args.sample_size), device=args.device)
        # adopt uniform sampling
        batch_num_nodes = [graph.number_of_nodes() for graph in batch_G]
        n_node = sum(batch_num_nodes)
        batch_perms = [[] for _ in range(args.sample_size)]
        batch_nodes = torch.ones(n_node)

        parameterizations = torch.split(batch_nodes, batch_num_nodes)
        for idx_g in range(len(batch_G)):
            # sample permutation
            batch_perm_g, ll_q_g, log_reps = sample_perm(graphs[idx_g]['G'], params=parameterizations[idx_g],
                                                              device=args.device, M=args.sample_size,
                                                              nobfs=args.nobfs, max_cr_iteration=args.max_cr_iteration)
            # batch_perms.append(perms)
            log_repetitions[idx_g].copy_(log_reps)
            ll_q[idx_g].copy_(ll_q_g)

            for idx_m in range(args.sample_size):
                batch_perms[idx_m].append(batch_perm_g[idx_m])
            # n_node = batch_G[b].number_of_nodes()
            # params = torch.ones(n_node).unsqueeze(0).expand(args.sample_size, n_node)
            # u = torch.distributions.utils.clamp_probs(torch.rand_like(params))
            # z = params - torch.log(-torch.log(u))
            # batch_perm_b = torch.sort(z, descending=True, dim=-1)[1]
            #
            # for b in range(real_batch_size):
            #     batch_perms[b].extend(batch_perm_b[b])

    log_repetitions = log_repetitions.to(args.device)

    # if args.enable_gcn:
    #     # do gcn sampling
    #     batch_nodes = gcn.forward(batch_dG, batch_dGX)  # (N, V*)
    # else:
    #     # do uniform sampling

    nll_p = torch.empty((real_batch_size, args.sample_size), device=args.device)
    if args.note == 'GraphRNN':
        # data process and training for graphRNN
        for m in range(args.sample_size):
            data = [processor(graph, perms) for graph, perms in zip(batch_G, batch_perms[m])]
            data = collate(data)
            nll_p_m = eval_loss_graph_rnn(args, model, data, feature_map)
            # nll_prob = eval_loss_graph_rnn(args, model, data, feature_map)
            nll_p[:, m].copy_(nll_p_m)
    if args.note == 'GRAN':
        # data process and training for graphRNN
        for m in range(args.sample_size):
            data = [processor(graph, perms) for graph, perms in zip(batch_G, batch_perms[m])]
            data = processor.collate_fn(data)
            nll_p_m = eval_loss_gran(args, model, data[0])
            # nll_prob = eval_loss_graph_rnn(args, model, data, feature_map)
            nll_p[:, m].copy_(nll_p_m)

    elif args.note == 'DGMG':
        # TODO: data process and training for DGMG
        # for i_g, graph in enumerate(batch_G):
            # data = [processor(graph, perms[i_g]) for perms in batch_perms]
            # # data = processor.collate_batch(data)
            # nll_p_g = eval_loss_dgmg(model['dgmg'], data)
            # nll_p[i_g, :].copy_(nll_p_g)
        assert args.batch_size == 1
        for m in range(args.sample_size):
            data = [processor(graph, perms) for graph, perms in zip(batch_G, batch_perms[m])]
            # data = processor.collate_batch(data)
            nll_p_m = eval_loss_dgmg(model['dgmg'], data)
            # nll_prob = eval_loss_graph_rnn(args, model, data, feature_map)
            nll_p[:, m].copy_(nll_p_m)

    if args.note == 'Graphgen':
        for m in range(args.sample_size):
            data = [processor(code, feature_map) for code in dfs_code_list[m::args.sample_size]]
            data = collate(data)
            nll_p_m = eval_loss_graphgen(args, model, data, feature_map)
            nll_p[:,m].copy_(nll_p_m)




    # log p_hat(G, pi) = log p(G|pi)p(pi) - log rep
    ll_p_hat = -nll_p - log_repetitions
    # print(ll_p_hat)
    # fake loss for q's gradient estimation
    if not args.enable_gcn:
        fake_nll_q = 0
    else:
        fake_nll_q = -torch.mean(torch.mean((ll_p_hat.detach()-ll_q.detach()) * ll_q, dim=1))

    nll_p = -torch.mean(torch.mean(ll_p_hat, dim=1))

    # compute elbo ( no gradient computation involved) elbo = 1/M * sum_{i=1}^M elbo_i
    elbo = torch.mean(ll_p_hat.detach()-ll_q.detach())

    ll_q = torch.mean(ll_q)
    # print(entropy)
    return nll_p, fake_nll_q, elbo, ll_q


def train_epoch(
        epoch, args, model, gcn, dataloader_train, processor, sample_perm,
        optimizer, scheduler, feature_map, log_history, summary_writer=None):
    # Set training mode for modules
    for _, net in model.items():
        net.train()

    if args.enable_gcn:
        gcn.train()

    batch_count = len(dataloader_train)
    total_loss = 0.0
    for batch_id, graphs in enumerate(dataloader_train):

        for _, net in model.items():
            net.zero_grad()

        if args.enable_gcn:
            gcn.zero_grad()

        st = time.time()
        nll_p, fake_nll_q, elbo, ll_q = evaluate_loss(args, model, gcn, processor, sample_perm, graphs, feature_map, epoch)
        loss = nll_p + fake_nll_q

        loss.backward()
        gradient = gcn.node_readout.FC_layers[0].bias.grad


        total_loss += elbo.data.item()

        spent = time.time() - st
        if batch_id % args.print_interval == 0:
            print('epoch {} batch {}: elbo is {}, llq is {}, time spent is {}.'.format(epoch, batch_id, elbo, ll_q, spent), flush=True)

        log_history['batch_elbo'].append(elbo.data.item())
        log_history['batch_time'].append(spent)

        # Update params of rnn and mlp
        for _, opt in optimizer.items():
            opt.step()

        for _, sched in scheduler.items():
            sched.step()

        if args.log_tensorboard:
            summary_writer.add_scalar('{} {} Loss/train batch'.format(
                args.note, args.graph_type), loss, batch_id + batch_count * epoch)

    return total_loss / batch_count


def test_data(epoch, args, model, gcn, dataloader_validate, processor, sample_perm, feature_map):
    for _, net in model.items():
        net.eval()
    gcn.eval()

    batch_count = len(dataloader_validate)
    with torch.no_grad():
        total_loss = 0.0
        ll_qs = 0.0
        for _, graphs in enumerate(dataloader_validate):
            loss_model, loss_gcn, elbo, ll_q = evaluate_loss(args, model, gcn, processor, sample_perm, graphs, feature_map, epoch)
            # loss = loss_model + loss_gcn
            # total_loss += loss.data.item()
            ll_qs += ll_q
            total_loss += elbo.data.item()

    return total_loss / batch_count, ll_qs


# Main training function

def train(args, model, gcn, feature_map, dataloader_train, dataloader_validate, processor, sample_perm):
    optimizer = {}
    for name, net in model.items():
        optimizer['optimizer_' + name] = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
            weight_decay=5e-5)
    if args.enable_gcn:
        optimizer['optimizer_gcn'] = optim.Adam(gcn.parameters(), lr=args.lr*5, weight_decay=5e-5)

    scheduler = {}
    for name, net in model.items():
        scheduler['scheduler_' + name] = MultiStepLR(
            optimizer['optimizer_' + name], milestones=args.milestones,
            gamma=args.gamma)

    if args.enable_gcn:
        scheduler['scheduler_gcn'] = MultiStepLR(
            optimizer['optimizer_gcn'], milestones=args.milestones,
            gamma=args.gamma)

    log_history = defaultdict(list)
    if args.load_model:

        fname, epoch = get_last_checkpoint(args, epoch=args.epochs_end)
        load_model(path=fname, device=args.device, model=model, gcn=gcn, optimizer=optimizer, scheduler=scheduler)
        print('Model loaded')
        df_iter = pd.read_csv(os.path.join(args.logging_iter_path))
        df_epoch = pd.read_csv(os.path.join(args.logging_epoch_path))

        log_history['batch_elbo'] = df_iter['batch_elbo'].tolist()
        log_history['batch_time'] = df_iter['batch_time'].tolist()
        log_history['train_elbo'] = df_epoch['train_elbo'].tolist()
        log_history['valid_elbo'] = df_epoch['valid_elbo'].tolist()

    else:
        epoch = 0

    if args.log_tensorboard:
        writer = SummaryWriter(
            log_dir=args.tensorboard_path+ ' ' + args.time, flush_secs=5)
    else:
        writer = None


    while epoch < args.epochs:
        # train
        loss = train_epoch(
            epoch, args, model, gcn, dataloader_train, processor, sample_perm, optimizer, scheduler, feature_map, log_history, writer)
        epoch += 1

        if args.log_tensorboard:
            writer.add_scalar('{} {} Loss/train'.format(args.note, args.graph_type), loss, epoch)

        print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

        # validate
        loss_validate, entropys_validate = test_data(epoch, args, model, gcn, dataloader_validate, processor, sample_perm, feature_map)
        entropys_validate = entropys_validate / args.sample_size
        if args.log_tensorboard:
            writer.add_scalar('{} {} Loss/validate'.format(args.note, args.graph_type), loss_validate, epoch)

        print('Epoch: {}/{}, validation loss: {:.6f}  entropy: {:.6f}'.format(epoch, args.epochs, loss_validate, entropys_validate), flush=True)

        # save model
        save_model(epoch, args, model, gcn, optimizer, scheduler, feature_map=feature_map)

        print('Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

        log_history['train_elbo'].append(loss)
        log_history['valid_elbo'].append(loss_validate)

        # save logging history
        df_iter = pd.DataFrame()
        df_epoch = pd.DataFrame()
        df_iter['batch_elbo'] = log_history['batch_elbo']
        df_iter['batch_time'] = log_history['batch_time']
        df_epoch['train_elbo'] = log_history['train_elbo']
        df_epoch['valid_elbo'] = log_history['valid_elbo']

        df_iter.to_csv(args.logging_iter_path, index=False)
        df_epoch.to_csv(args.logging_epoch_path, index=False)

