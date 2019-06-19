import torch
import itertools
from collections import defaultdict


#### below functions let you provide edges
def make_log_potentials(T, K, V, bsz, edges):
    """
    edges - nedges x 2
    T, K, V = seqlength, nstates, vocab
    returns
      nedges x bsz x K x K, T x bsz x K x V
    """
    # make all pairwise log potentials
    pwlpots = torch.randn(edges.size(0), bsz, K, K)
    # make all emission log dists
    emlps = torch.log_softmax(torch.randn(T*bsz*K, V), dim=1).view(T, bsz, K, -1)
    return pwlpots, emlps


# this gives real unary factors and doesn't normalize anything
def make_rbmlike_log_potentials(T, K, bsz, edges):
    """
    edges - nedges x 2
    T, K, V = seqlength, nstates, vocab
    returns
      nedges x bsz x K x K, T x bsz x K
    """
    # make all pairwise log potentials
    pwlpots = torch.randn(edges.size(0), bsz, K, K)
    # make all emission log dists
    emlps = torch.randn(T, bsz, K)
    return pwlpots, emlps

def joint_energy(x, z, pwlpots, emlps, edges):
    """
    not normalized.
    x, z - T x bsz, T x bsz
    """
    assert x.size() == z.size()
    T, bsz = x.size()
    _, _, K, V = emlps.size()
    jen = torch.zeros(bsz)
    for c, row in enumerate(edges):
        left, right = row
        left, right = left.item(), right.item()
        for b in range(bsz):
            llab, rlab = z[left, b].item(), z[right, b].item()
            jen[b] += pwlpots[c][b, llab, rlab]
    statedists = emlps.view(-1, K, V)[torch.arange(T*bsz), z.view(-1)] # T*bsz x V
    # # can also do:
    # statedists = emlps.view(-1, K, V).gather(1, z.view(-1, 1, 1).expand(T*bsz, 1, V)).squeeze()
    lps = statedists.gather(1, x.view(-1, 1)) # T*bsz x 1
    jen.add_(lps.view(T, bsz).sum(0))
    return jen


def brute_joint_partition(pwlpots, emlps, edges):
    T, bsz, K, V = emlps.size()
    jens = torch.zeros(bsz, K**T * V**T)
    idx = 0
    for z in itertools.product(range(K), repeat=T):
        for x in itertools.product(range(V), repeat=T):
            xtens = torch.LongTensor(x).view(T, 1).expand(T, bsz).contiguous()
            ztens = torch.LongTensor(z).view(T, 1).expand(T, bsz).contiguous()
            jen = joint_energy(xtens, ztens, pwlpots, emlps, edges)
            jens[:, idx] = jen
            idx += 1
    Zs = torch.logsumexp(jens, dim=1)
    return Zs


def brute_z_partition(pwlpots, T, edges):
    """
    edges are nedges x 2
    """
    # Because x distribution is locally normalized, this should give the same thing as above!
    _, bsz, K, _ = pwlpots.size()
    jens = torch.zeros(bsz, K**T)
    idx = 0
    for z in itertools.product(range(K), repeat=T):
        for c, row in enumerate(edges):
            left, right = row
            left, right = left.item(), right.item()
            for b in range(bsz):
                llab, rlab = z[left], z[right]
                jens[b, idx] += pwlpots[c][b, llab, rlab]
        idx += 1
    Zs = torch.logsumexp(jens, dim=1)
    return Zs


def brute_z_fac_lmarg(pwlpots, T, edges, tsrc, ttgt, x=None, emlps=None):
    """
    edges are nedges x 2
    returns bsz x K x K unnormalized log marginals for edge tsrc, ttgt
    """
    # Because x distribution is locally normalized, this should give the same thing as above!
    _, bsz, K, _ = pwlpots.size()
    jens = torch.zeros(bsz, K, K, K**(T-2))
    idxs = torch.LongTensor(K, K).zero_()
    for z in itertools.product(range(K), repeat=T):
        tsval, ttval = z[tsrc], z[ttgt]
        idx = idxs[tsval, ttval]
        if emlps is not None:
            ztens = torch.LongTensor(z).view(T, 1).expand(T, bsz).contiguous()
            jens[:, tsval, ttval, idx] += joint_energy(x, ztens, pwlpots, emlps, edges)
        else:
            for c, row in enumerate(edges):
                left, right = row
                left, right = left.item(), right.item()
                for b in range(bsz):
                    llab, rlab = z[left], z[right]
                    jens[b, tsval, ttval, idx] += pwlpots[c][b, llab, rlab]
        idxs[tsval, ttval] += 1
    faclmarg = torch.logsumexp(jens, dim=3)
    return faclmarg


def brute_z_marg(x, pwlpots, emlps, edges):
    T, bsz, K, V = emlps.size()
    jens = torch.zeros(bsz, K**T)
    idx = 0
    for z in itertools.product(range(K), repeat=T):
        ztens = torch.LongTensor(z).view(T, 1).expand(T, bsz).contiguous()
        jen = joint_energy(x, ztens, pwlpots, emlps, edges)
        jens[:, idx] = jen
        idx += 1
    Zs = torch.logsumexp(jens, dim=1)
    return Zs


def init_msgs(bsz, K, edges, device, x=None, emlps=None):
    """
    inits messages and other stuff
    x - T x bsz
    emlps - T x bsz x K x V or T x bsz x K
    """
    assert emlps is None or emlps.size(1) == bsz
    # if x is None them emlps must already have stuff selected
    assert emlps is None or x is not None or emlps.dim() == 3

    fmsgs, nmsgs, nodene = {}, {}, defaultdict(list)
    for e, edge in enumerate(edges):
        s, t = edge[0].item(), edge[1].item()
        fmsgs[e, s] = torch.zeros(bsz, K).to(device)
        fmsgs[e, t] = torch.zeros(bsz, K).to(device)
        nmsgs[s, e] = torch.zeros(bsz, K).to(device)
        nmsgs[t, e] = torch.zeros(bsz, K).to(device)
        nodene[s].append(e)
        nodene[t].append(e)

    if emlps is not None: # will add a bunch of factor msgs
        for i in range(emlps.size(0)):
            omsg = torch.Tensor(bsz, K).to(device)
            if emlps.dim() == 3:
                omsg.copy_(emlps[i])
            else:
                for b in range(bsz):
                    omsg[b].copy_(emlps[i][b][:, x[i][b]])
            fmsgs[edges.size(0)+i, i] = omsg
            nodene[i].append(edges.size(0)+i)
    return nmsgs, fmsgs, nodene


def get_beliefs(nmsgs, fmsgs, pwlpots, edges):
    _, bsz, K, _ = pwlpots.size()
    nbeliefs, fbeliefs = {}, {}
    for e, edge in enumerate(edges):
        s, t = edge[0].item(), edge[1].item()
        fbeliefs[e] = pwlpots[e] + nmsgs[s, e].unsqueeze(2) + nmsgs[t, e].unsqueeze(1)
        if s not in nbeliefs:
            nbeliefs[s] = pwlpots.new(bsz, K).zero_()
        if t not in nbeliefs:
            nbeliefs[t] = pwlpots.new(bsz, K).zero_()
        nbeliefs[s] += fmsgs[e, s]
        nbeliefs[t] += fmsgs[e, t]

    # check if we have unary factor messages
    unary_fmsgs = [(ue, s) for (ue, s) in fmsgs.keys() if ue >= edges.size(0)]
    for (ue, s) in unary_fmsgs: # only nonempty if we have unary factors
        nbeliefs[s] += fmsgs[ue, s]

    return nbeliefs, fbeliefs

# we'll associate a factor w/ each edge
def bp_update(src, dst, nmsgs, fmsgs, pwlpots, edges, nodene,
              node_msg=True, renorm=False):
    if node_msg: # src is a node index and dst is a factor index
        # get all messages from factor neighbors other than dst
        neighbfmsgs = [fmsgs[facne, src] for facne in nodene[src] if facne != dst]
        if len(neighbfmsgs) > 0:
            #nmsgs[src, dst] = sum(neighbfmsgs)
            numsg = sum(neighbfmsgs)
            if renorm:
                numsg = torch.log_softmax(numsg, dim=1)
            diffnorm = torch.norm(nmsgs[src, dst] - numsg).item()
            nmsgs[src, dst] = numsg
        else:
            diffnorm = 0
    else: # src is a factor index and dst is a node index
        lpots = pwlpots[src] # bsz x K x K
        # get all messages from node neighbors other than dst; since we're only doing pw factors,
        # there's only one
        fleft, fright = edges[src][0].item(), edges[src][1].item()
        if fright == dst: # moving "left to right" along the edge
            #fmsgs[src, dst] = torch.logsumexp(lpots + nmsgs[fleft, src].unsqueeze(2), dim=1)
            numsg = torch.logsumexp(lpots + nmsgs[fleft, src].unsqueeze(2), dim=1)
        else: # moving "right to left" along the edge
            numsg = torch.logsumexp(lpots + nmsgs[fright, src].unsqueeze(1), dim=2)
        if renorm:
            numsg = torch.log_softmax(numsg, dim=1)
        diffnorm = torch.norm(fmsgs[src, dst] - numsg).item()
        fmsgs[src, dst] = numsg
    return diffnorm

# this just walks the edges back and forth
def dolbp(pwlpots, edges, x=None, emlps=None, niter=1, renorm=False, tol=1e-3, randomize=False):
    """
    pwlpots - nedges x bsz x K x K
    edges - nedges x 2
    x - T x bsz
    emlps - T x bsz x K x V or T x bsz x K
    """
    nedges, bsz, K, _ = pwlpots.size()
    nmsgs, fmsgs, nodene = init_msgs(bsz, K, edges, pwlpots.device, x=x, emlps=emlps)
    if renorm:
        for thing in nmsgs.keys():
            nmsgs[thing] = torch.log_softmax(nmsgs[thing], dim=1)
        for thing in fmsgs.keys():
            fmsgs[thing] = torch.log_softmax(fmsgs[thing], dim=1)

    # make a list of edge indexes and whether left to right
    order = [(e, True) for e in range(nedges)] + [(e, False) for e in range(nedges-1, -1, -1)]
    iters_taken = niter
    for i in range(niter):
        avgdiffnorm = 0.0
        if randomize:
            perm = torch.randperm(len(order))
            order = [order[idx.item()] for idx in perm]

        for e, left_to_right in order:
            s, t = edges[e][0].item(), edges[e][1].item()
            if left_to_right:
                # send a node message
                dnorm = bp_update(s, e, nmsgs, fmsgs, pwlpots, edges, nodene,
                                  node_msg=True, renorm=renorm)
                avgdiffnorm += dnorm
                # send a factor message
                dnorm = bp_update(e, t, nmsgs, fmsgs, pwlpots, edges, nodene,
                                  node_msg=False, renorm=renorm)
                avgdiffnorm += dnorm
            else:
                # send a node message
                dnorm = bp_update(t, e, nmsgs, fmsgs, pwlpots, edges, nodene,
                                  node_msg=True, renorm=renorm)
                avgdiffnorm += dnorm
                # send a factor message
                dnorm = bp_update(e, s, nmsgs, fmsgs, pwlpots, edges, nodene,
                                  node_msg=False, renorm=renorm)
                avgdiffnorm += dnorm

        if avgdiffnorm/(len(order) * 2.0) <= tol:
            iters_taken = i+1
            break

    # make beliefs
    nbeliefs, fbeliefs = get_beliefs(nmsgs, fmsgs, pwlpots, edges)
    # normalize them
    for k in nbeliefs.keys():
        nbeliefs[k] = nbeliefs[k].log_softmax(1)
    for k in fbeliefs.keys():
        fbeliefs[k] = fbeliefs[k].view(bsz, -1).log_softmax(1).view(bsz, K, K)

    return nbeliefs, fbeliefs, iters_taken, nmsgs, fmsgs
