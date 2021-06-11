import torch
import torch.nn.functional as F
import torch.multiprocessing as multiprocessing
import itertools
from torch._six import queue
from torch.utils.data._utils.worker import ManagerWatchdog
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL, python_exit_status
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from networkx.algorithms import isomorphism
import pynauty as pnt
from utils import nx_to_nauty




def logcumsumexp(x, dim):
    # slow implementation, but ok for now
    if (dim != -1) or (dim != x.ndimension() - 1):
        x = x.transpose(dim, -1)
    out = []
    for i in range(1, x.size(-1) + 1):
        out.append(torch.logsumexp(x[..., :i], dim=-1, keepdim=True))
    out = torch.cat(out, dim=-1)
    if (dim != -1) or (dim != x.ndimension() - 1):
        out = out.transpose(-1, dim)
    return out


def smart_perm(x, permutation):
    assert x.size() == permutation.size()
    if x.ndimension() == 1:
        ret = x[permutation]
    elif x.ndimension() == 2:
        d1, d2 = x.size()
        ret = x[
            torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
            permutation.flatten()
        ].view(d1, d2)
    elif x.ndimension() == 3:
        d1, d2, d3 = x.size()
        ret = x[
            torch.arange(d1).unsqueeze(1).repeat((1, d2 * d3)).flatten(),
            torch.arange(d2).unsqueeze(1).repeat((1, d3)).flatten().unsqueeze(0).repeat((1, d1)).flatten(),
            permutation.flatten()
        ].view(d1, d2, d3)
    else:
        ValueError("Only 3 dimensions maximum")
    return ret


def reverse_logcumsumexp(x, dim):
    return torch.flip(logcumsumexp(torch.flip(x, dims=(dim, )), dim), dims=(dim, ))


class PlackettLuce(Distribution):
    """
        Plackett-Luce distribution
    """
    arg_constraints = {"logits": constraints.real}
    def __init__(self, logits):
        # last dimension is for scores of plackett luce
        super(PlackettLuce, self).__init__()
        self.logits = logits
        self.size = self.logits.size()

    def sample(self, num_samples):
        # sample permutations using Gumbel-max trick to avoid cycles
        with torch.no_grad():
            logits = self.logits.unsqueeze(0).expand(num_samples, *self.size)
            u = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
            z = self.logits - torch.log(-torch.log(u))
            samples = torch.sort(z, descending=True, dim=-1)[1]
        return samples

    def log_prob(self, samples):
        # samples shape is: num_samples x self.size
        # samples is permutations not permutation matrices
        if samples.ndimension() == self.logits.ndimension():  # then we already expanded logits
            logits = smart_perm(self.logits, samples)
        elif samples.ndimension() > self.logits.ndimension():  # then we need to expand it here
            logits = self.logits.unsqueeze(0).expand(*samples.size())
            logits = smart_perm(logits, samples)
        else:
            raise ValueError("Something wrong with dimensions")
        logp = (logits - reverse_logcumsumexp(logits, dim=-1)).sum(-1)
        return logp


def color_refinement(G, iterations, edge_attr=None, node_attr=None):
    """
    This is adopted from weisfeiler_lehman_graph_hash function in networkx
    """
    def neighborhood_aggregate(G, node, node_labels, edge_attr=None):
        """
        Compute new labels for given node by aggregating
        the labels of each node's neighbors.
        """
        label_list = [node_labels[node]]
        for nei in G.neighbors(node):
            prefix = "" if not edge_attr else G[node][nei][edge_attr]
            label_list.append(prefix + node_labels[nei])
        return "".join(sorted(label_list))

    def weisfeiler_lehman_step(G, labels, edge_attr=None, node_attr=None):
        """
        Apply neighborhood aggregation to each node
        in the graph.
        Computes a dictionary with labels for each node.
        """
        new_labels = dict()
        for node in G.nodes():
            new_labels[node] = neighborhood_aggregate(
                G, node, labels, edge_attr=edge_attr
            )
        return new_labels

    node_labels = dict()

    # set initial node labels
    for node in G.nodes():
        if (not node_attr) and (not edge_attr):
            node_labels[node] = str(G.degree(node))
        elif node_attr:
            node_labels[node] = str(G.nodes[node][node_attr])
        else:
            node_labels[node] = ""
    for k in range(iterations):
        node_labels = weisfeiler_lehman_step(G, node_labels, edge_attr=edge_attr)

    return node_labels


def compute_repetition(graph, perm, device, max_cr_iteration):
    # compute repetition given a graph and order
    subgraphs = [graph.subgraph(perm[:i+1]) for i in range(len(perm))]
    reps = []
    # cr_time = []
    for i, subgraph in enumerate(subgraphs):
        # TODO: add early stop
        # st = time.time()
        node_labels = color_refinement(subgraph, iterations=min(subgraph.number_of_nodes(), max_cr_iteration)) # use 10 here to avoid expensive computation!
        # cr_time.append(time.time() - st)
        reps.append(list(node_labels.values()).count(node_labels[perm[i]]))
    # print(reps)
    # print(cr_time)
    log_rep = torch.sum(torch.log(torch.tensor(reps, dtype=torch.float32, requires_grad=False, device=device)))
    return log_rep


def sample_legal_perm(graph, params, device):
    perm = []
    added_nodes = set()
    candidates = set()
    log_q_prob = torch.tensor(0, dtype=torch.float32, device=device)
    categorical_params = F.softmax(params, dim=0)

    start_node = torch.multinomial(categorical_params, 1).item()
    perm.append(start_node)
    added_nodes.add(start_node)
    candidates.update([n for n in graph.neighbors(start_node)])
    log_q_prob += torch.log(categorical_params[start_node])

    while len(perm) != params.shape[0]:
        l_candidates = list(candidates)
        categorical_params = F.softmax(params[l_candidates], dim=0)
        cur_node = l_candidates[torch.multinomial(categorical_params, 1).item()]
        perm.append(cur_node)
        added_nodes.add(cur_node)
        candidates.update([n for n in graph.neighbors(cur_node)])
        candidates.difference_update(added_nodes)
        log_q_prob += torch.log(categorical_params[l_candidates.index(cur_node)])

    assert len(candidates) == 0, "still have nodes left in the candidate list!"

    return perm, log_q_prob


# TODO: modify bfs sampler and keep it consistent 'sample_legal_perm' and pleaset TEST it! @Xu
def sample_legal_bfsperm(graph, params, device):
    perm = []
    added_nodes = set()
    candidates = set()
    log_q_prob = torch.tensor(0, dtype=torch.float32, device=device)
    categorical_params = F.softmax(params, dim=0)
    # start_node = torch.multinomial(categorical_params.detach(), 1).item()
    start_node = torch.multinomial(categorical_params, 1).item()
    perm.append(start_node)
    added_nodes.add(start_node)
    candidates.update([n for n in graph.neighbors(start_node)])
    log_q_prob += torch.log(categorical_params[start_node])

    while len(perm) != params.shape[0]:
        l_candidates = list(candidates)
        categorical_params = F.softmax(params[l_candidates], dim=0)
        cur_node = l_candidates[torch.multinomial(categorical_params, 1).item()]
        perm.append(cur_node)
        added_nodes.add(cur_node)
        candidates.update([n for n in graph.neighbors(cur_node)])
        candidates.difference_update(added_nodes)
        log_q_prob += torch.log(categorical_params[l_candidates.index(cur_node)])

    assert len(candidates) == 0, "still have nodes left in the candidate list!"

    return perm, log_q_prob


# Single-processing version
def legal_perms_sampler(graph, params, device, M=1, nobfs=True, max_cr_iteration=10):
    '''
    :param graph: networkx graph
    :param params: params (N, )
    :param M: number of samples
    :return: sequences and the repetitions of each
    '''
    # params = params.detach()
    perms = []
    log_reps = torch.empty(M)
    log_probs = torch.empty(M)

    for m in range(M):
        if nobfs:
            # st = time.time()
            perm, log_prob = sample_legal_perm(graph, params, device)
            # t_sample_legal_perm+= time.time()-st
            # st = time.time()
            rep = compute_repetition(graph, perm, device, max_cr_iteration)
            # t_compute_repetition+= time.time()-st
            perms.append(perm)
            log_reps[m].fill_(rep)
            log_probs[m].fill_(log_prob)
        else:
            perm, log_prob = sample_legal_bfsperm(graph, params, device)
            rep = compute_repetition(graph, perm, device, max_cr_iteration)
            perms.append(perm)
            log_reps[m].fill_(rep)
            log_probs[m].fill_(log_prob)
    # print("sample: ", t_sample_legal_perm, "rep: ", t_compute_repetition)
    return perms, log_probs, log_reps


# Multi-processing version
def worker_loop(index_queue, data_queue, done_event):
    try:
        iteration_end = False

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if r is None:
                # Received the final signal
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                continue
            subgraph, iterations, node_id = r
            # try:

            node_labels = color_refinement(subgraph, iterations)
            rep = list(node_labels.values()).count(node_labels[node_id])
            # rep = color_refinement(subgraph, iterations, node_id)
            # except Exception as e:
            #     rep = 1
            data_queue.put(rep)
            del iterations, node_id, subgraph, node_labels  # save memory
    except KeyboardInterrupt:
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()

def worker_loop_vf2(index_queue, data_queue, done_event):
    try:
        iteration_end = False

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if r is None:
                # Received the final signal
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                continue
            subgraph, iterations, node_id = r
            # try:

            node_labels = color_refinement(subgraph, iterations)
            iso_nodes = [node for node, label in node_labels.items() if label == node_labels[node_id]]
            # print(len(iso_nodes))
            iso_nodes.remove(node_id)
            nodes = list(subgraph.nodes())
            target_nodes = nodes[:]
            target_nodes.remove(node_id)
            target_graph = subgraph.subgraph(target_nodes)
            rep = 1
            for iso_node in iso_nodes:
                source_nodes = nodes[:]
                source_nodes.remove(iso_node)
                source_graph = subgraph.subgraph(source_nodes)
                GM = isomorphism.GraphMatcher(target_graph, source_graph).is_isomorphic()
                if GM:
                    rep += 1
            # rep = color_refinement(subgraph, iterations, node_id)
            # except Exception as e:
            #     rep = 1
            data_queue.put(rep)
            del iterations, node_id, subgraph, node_labels, target_nodes, target_graph  # save memory
    except KeyboardInterrupt:
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()

def compute_autogrp_n(graph):
    na_graph = nx_to_nauty(graph)
    n_autogrp = pnt.autgrp(na_graph)[1]
    return torch.log(torch.tensor(n_autogrp, dtype=torch.float32, requires_grad=False))

class mp_sampler():
    def __init__(self, args, vf2=False):
        self.args = args
        self._workers = []
        self._index_queues = []
        self.device = args.device
        self._worker_result_queue = multiprocessing.Queue()
        self._workers_done_event = multiprocessing.Event()
        self._num_workers = args.mp_num_workers
        self.max_cr_iteration = args.max_cr_iteration
        wkl = worker_loop_vf2 if vf2 else worker_loop

        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing.Queue()  # type: ignore
            # index_queue.cancel_join_thread()
            w = multiprocessing.Process(
                target=wkl,
                args=(index_queue, self._worker_result_queue, self._workers_done_event))
            w.daemon = True
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

    def compute_repetition(self, graph, perm):
        perm_length = len(perm)
        # subgraphs = ([graph.subgraph(perm[:i + 1]) for i in range(len(perm))])
        # for idx in range(perm_length):
        for idx in range(perm_length-1, -1, -1):
            # TODO: add early stop
            worker_id = idx % self._num_workers
            subgraph = graph.subgraph(perm[:idx + 1])
            self._index_queues[worker_id].put((subgraph, min(subgraph.number_of_nodes(), self.max_cr_iteration), perm[idx]))
        count = 0
        results = []
        while count != perm_length:
            results.append(self._worker_result_queue.get(timeout=MP_STATUS_CHECK_INTERVAL))
            count += 1
        self._worker_result_queue.empty()
        results = torch.tensor(results, dtype=torch.float32, requires_grad=False)
        log_rep = torch.sum(torch.log(results))
        return log_rep

    def __call__(self, graph, params, device, M=1, nobfs=True, max_cr_iteration=10):
        self.max_cr_iteration = max_cr_iteration
        self.device = device
        # perms = []
        # log_probs = torch.empty(M, device=device)

        # perm, log_prob = self.sample_legal_perm(graph, params)
        log_reps = torch.empty(M)
        perms = PlackettLuce(logits=params).sample(M)
        log_probs = PlackettLuce(logits=params).log_prob(perms)
        perms = perms.tolist()
        for m in range(M):
            if nobfs:
                rep = self.compute_repetition(graph, perms[m])
                log_reps[m].fill_(rep)
            else:

                # TODO: rewrite bfs sampler as a class function @Xu
                pass
        return perms, log_probs, log_reps

    def _shutdown_workers(self):
        if python_exit_status is True or python_exit_status is None:
            return
        self._workers_done_event.set()
        for w in self._workers:
            w.join(timeout=MP_STATUS_CHECK_INTERVAL)
            if w.is_alive():
                w.terminate()
        for q in self._index_queues:
            q.cancel_join_thread()
            q.close()

    def __del__(self):
        self._shutdown_workers()


# if __name__ == '__main__':
    # # TODO: 1. test the sample graph is always connected
    # #       2. test the repetition computation is correct
    # #       3. test the probability computation is correct
    # #       4. improve the computing efficiency in func "compute_repetition"
    # import networkx as nx
    # from args import Args
    # args = Args()
    # MPS = mp_sampler(args)
    # import time
    # N = 50
    # graph = nx.path_graph(N)
    # params = list(range(N))
    # ct = 0
    # for i in range(100):
    #     st = time.time()
    #     s = MPS.compute_repetition(graph, params)
    #     ct += time.time()-st
    #     print(s)
    # print(ct/100) #time: 0.7-200; 0.047-50; 4.3-500

# if __name__ == '__main__':
#     import networkx as nx
#     graph = nx.grid_2d_graph(5, 5)
#     nodes = list(graph.nodes())
#     mapping = {nodes[i]: i for i in range(len(nodes))}
#     graph = nx.relabel_nodes(graph, mapping)
#     pass
    # import networkx as nx
    # from networkx.algorithms import isomorphism
    #
    # N =32
    # graph = nx.path_graph(N)
    #
    # #perms = list(map(list, itertools.permutations(list(range(N)))))
    # node_labels = color_refinement(graph, 3)
    # target = 4
    # iso_nodes = [node for node, label in node_labels.items() if label == node_labels[target]]
    #
    # print(len(iso_nodes))
    # iso_nodes.remove(target)
    # nodes = list(graph.nodes())
    # target_nodes = nodes[:]
    # target_nodes.remove(target)
    # target_graph = graph.subgraph(target_nodes)
    # rep = 1
    # for iso_node in iso_nodes:
    #     source_nodes = nodes[:]
    #     source_nodes.remove(iso_node)
    #     source_graph = graph.subgraph(source_nodes)
    #     GM = isomorphism.GraphMatcher(target_graph, source_graph).is_isomorphic()
    #     if GM:
    #         rep += 1
    # print(rep)
    # pass
    # params = list(range(N))

    # ct = 0
#
#     for i in range(100):
#         st = time.time()
#         s = compute_repetition(graph, params, 'cpu', 10)
#         ct += time.time()-st
#         print(s)
#     print(ct/100) # time: 1.6411763548851013-200; 0.14-50; 10-500
