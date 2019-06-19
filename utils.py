import math
import torch
import numpy as np

def corr(t1, t2):
    return np.corrcoef(t1.data.cpu().numpy(), t2.data.cpu().numpy())[0][1]

def get_hmm_ne(M, T, K):
    # get number of factor neighbors -1 for each node
    order = min(M, T-1)
    ne = torch.Tensor(T).fill_(2*order-1) # order behind + order in front
    for j in range(order): # handle edge cases
        ne[j] -= (order -j)
        ne[T-j-1] -= (order -j)
    # # observations add another factor per node
    # ne_x = ne + 1
    # finally we repeat these per value of K just by convention
    ne = ne.view(T, 1).repeat(1, K) # T x K
    #ne_x = ne_x.view(T, 1).repeat(1, K) # T x K
    return ne

def get_hmm_edges(T, M):
    """
    returns nedges x 2 tensor of indices
    """
    order = min(M, T-1)
    edges = torch.stack([torch.LongTensor([i, j])
                         for i in range(T-1) for j in range(i+1, min(i+order+1, T))])
    return edges

def get_hmm_stuff(T, M, K):
    edges = get_hmm_edges(T, M)
    nedges = edges.size(0)
    # rowidxs are T x max outgoing edges
    nodeidxs = [[] for _ in range(T)]
    for i, (s, t) in enumerate(edges):
        nodeidxs[s.item()].append(i)
        nodeidxs[t.item()].append(nedges + i)
    # pad as necessary
    maxne = max(len(nez) for nez in nodeidxs)
    for t in range(T):
        nodeidxs[t].extend([2*nedges] * (maxne - len(nodeidxs[t])))
    nodeidxs = torch.LongTensor(nodeidxs)
    ne = get_hmm_ne(M, T, K)
    return edges, nodeidxs, ne



def get_rbm_edges(V, H):
    edges = torch.stack([torch.LongTensor([i, V+j])
                         for i in range(V) for j in range(H)])
    return edges

# note this ignores unary factors, which we need to add back in
def get_rbm_ne(V, H):
    ne = torch.zeros(V+H)
    ne[:V] = (H-1)
    ne[V:] = (V-1)
    return ne.view(1, V+H, 1).expand(1, V+H, 2)

def batch_kl(p, q):
    """
    p and q are both the same size, and last dim has the K log probabilities
    returns bsz vector
    """
    return (p.exp()*(p - q)).sum(-1)


def unary_from_pw(tau, edges, T):
    """
    tau - bsz x nedges*K*K
    edges - nedges x 2
    returns bsz x nedges*K
    """
    bsz = tau.size(0)
    K = int(math.sqrt(tau.size(1)/edges.size(0)))
    bytab = tau.view(bsz, -1, K, K).transpose(0, 1) # nedges x bsz x K x K
    rowmargs = bytab.sum(3) # nedges x bsz x K
    colmargs = bytab.sum(2) # nedges x bsz x K
    nmargs = [0]*T
    denom = tau.new(T).zero_()
    for n, (s, t) in enumerate(edges):
        s, t = s.item(), t.item()
        nmargs[s] = nmargs[s] + rowmargs[n]
        denom[s] += 1
        nmargs[t] = nmargs[t] + colmargs[n]
        denom[t] += 1
    # take averages: T x bsz x K -> bsz x T x K
    nmargs = torch.stack(nmargs).transpose(0, 1) / denom.view(1, T, 1)
    return nmargs.contiguous().view(bsz, -1)


def bethe_fex(tau_u, tau_e, un_lpot, ed_lpot, ne):
    """
    tau_u - bsz x nnodes*K marginals
    tau_e - bsz x nedges*K*K marginals
    un_lpot - bsz x nnodes*K log potentials
    ed_lpot - bsz x nedges*K*K log potentials
    ne - bsz x nnodes*K number of neighbors-1 for each node (repeated), not incl obs factors
    """
    assert tau_u.size(0) == un_lpot.size(0)
    assert tau_e.size(0) == ed_lpot.size(0)
    assert tau_u.size(0) == ne.size(0)
    en = -(tau_u*un_lpot).sum() - (tau_e*ed_lpot).sum()
    negfacent = (tau_e*tau_e.log()).sum() #+ extra_ent*(b_n*b_n.log()).sum()
    nodent = (ne*tau_u*tau_u.log()).sum()
    #print("fex", en.item(), negfacent.item(), nodent.item())
    return en + negfacent - nodent, en, negfacent, nodent


def bethe_fez(tau_u, tau_e, ed_lpot, ne):
    """
    tau_u - 1 x nnodes*K marginals
    tau_e - 1 x nedges*K*K marginals
    ed_lpot - 1 x nedges*K*K log potentials
    ne - 1 x nnodes*K number of neighbors-1 for each node (repeated), not incl obs factors
    """
    assert tau_e.size(0) == ed_lpot.size(0)
    assert tau_u.size(0) == ne.size(0)
    en = -(tau_e*ed_lpot).sum()
    negfacent = (tau_e*tau_e.log()).sum()
    nodent = (ne*tau_u*tau_u.log()).sum()
    #print("fez", en.item(), negfacent.item(), nodent.item())
    return en + negfacent - nodent, en, negfacent, nodent


def clip_opt_params(optalg, max_norm):
    """
    would be strange if there were actually multiple param groups
    """
    for group in optalg.param_groups:
        torch.nn.utils.clip_grad_norm_(group['params'], max_norm)


class InfcHelper(object):
    """
    just for sequences now
    """
    def __init__(self, opt):
        self.markov_order = opt.markov_order
        self.K = opt.K # range of discrete values
        self.edge_cache = [None]*(opt.max_len+1)
        self.ending_at_cache = [None]*(opt.max_len+1)

    def get_edges(self, T):
        """
        returns nedges x 2 tensor of indices
        """
        if self.edge_cache[T] is None:
        # if T not in self.edge_cache:
            order = min(self.markov_order, T-1)
            edges = torch.stack([torch.LongTensor([i, j])
                                 for i in range(T-1) for j in range(i+1, min(i+order+1, T))])
            self.edge_cache[T] = edges
        return self.edge_cache[T]

    def get_ending_at(self, T):
        """
        returns map: t -> [idx s.t. edges[idx] = (s, t)] in ascending order of s
        """
        if self.ending_at_cache[T] is None:
        #if T not in self.ending_at_cache:
            ending_at = {t: [] for t in range(1, T)}
            edges = self.get_edges(T)
            for n in range(edges.size(0)):
                start, end = edges[n]
                start, end = start.item(), end.item()
                ending_at[end].append(n)
            self.ending_at_cache[T] = ending_at
        return self.ending_at_cache[T]
