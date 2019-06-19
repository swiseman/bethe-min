import math
import itertools
import torch
from collections import defaultdict

def _multi_idx_loop(A, idxs):
    """
    idxs a list
    """
    curr = A[idxs[0].item()]
    for i in range(1, len(idxs)):
        curr = curr[idxs[i].item()]
    return curr

def _multi_idx_flatten(A, idxs):
    """
    A - a d-dim tensor
    idxs - a d-length LongTensor
    """
    dims = torch.tensor(A.size())
    offsets = dims.flip(0).cumprod(0).flip(0)
    # calculate row in flattented mat
    idx = idxs[:-1].dot(offsets[1:]) + idxs[-1]
    return A.view(-1)[idx]

def _multi_idx_hack(A, idxs):
    """
    idxs now just a list
    """
    return eval("A%s" % idxs)

def gen_hmm_data(T=10, K=3, V=5, M=1, N=10, seed=3435, neural=False):
    """
    let's assume homogeneous for now?
    M - markov order
    we'll condition on T I guess
    """
    init = 1 if not neural else 0.1
    torch.manual_seed(seed)
    # make params
    tdims = [K+1]*(M) # extra class for start
    tdims.append(K)
    trans = torch.log_softmax(torch.Tensor(*tdims).uniform_(-init, init), dim=M)
    ems = torch.log_softmax(torch.Tensor(K, V).uniform_(-init, init), dim=1)

    exptrans = trans.exp()
    expems = ems.exp()
    data = []
    for _ in range(N):
        hist = [K]*M
        x = []
        for _ in range(T):
            z_t = torch.multinomial(_multi_idx(exptrans, hist), num_samples=1).item()
            x_t = torch.multinomial(expems[z_t], num_samples=1).item()
            x.append(x_t)
            hist = hist[1:] + [z_t]
        data.append(x)
    return data, trans, ems


def joint_lp(x, z, trans, ems):
    T, K = len(x), ems.size(0)
    M = trans.dim() - 1
    hist = [K]*M
    lp = 0
    for t in range(T):
        z_t = z[t]
        lp += (_multi_idx_loop(trans, hist)[z_t].item() + ems[z_t][x[t]].item())
        hist = hist[1:] + [z_t]
    return lp


def brute_marg(x, ems, trans):
    T, K = len(x), ems.size(0)
    lps = [joint_lp(x, z, trans, ems) for z in itertools.product(range(K), repeat=T)]
    return torch.logsumexp(torch.tensor(lps), dim=0)


def var_elim(x, ems, trans):
    T, K = len(x), ems.size(0)
    M = trans.dim() - 1
    # make possible histories for each position
    fullhists = [list(prod) for prod in itertools.product(range(K), repeat=M)]
    allhists = []
    for m in range(M):
        mhists = [[K]*(M-m)] # by convention
        if m > 0:
            rems = [list(prod) for prod in itertools.product(range(K), repeat=m)]
            first = mhists[0]
            mhists = [first + rem for rem in rems]
        allhists.append(mhists)
    # have full history for everyone else
    [allhists.append(fullhists) for _ in range(M, T)]
    # initialize base case
    tau = {tuple(hist): 0 for hist in allhists[T-1]}
    for t in range(T-1, -1, -1):
        nutau = {}
        for hist in allhists[t]: # sequence of length M
            terms = [_multi_idx_loop(trans, hist)[k].item() + ems[k][x[t]].item()
                     + tau[tuple(hist[1:] + [k])] for k in range(K)]
            nutau[tuple(hist)] = torch.logsumexp(torch.tensor(terms), dim=0)
        tau = nutau
    assert len(tau) == 1
    return tau[tuple(allhists[0][0])]


HISTCACHE = {} # K, m -> list of possible histories

def get_hists(K, m, M=None, start=False):
    """
    assumes will only need to cache for one M (i.e., markov order)
    """
    if (K, m, start) not in HISTCACHE:
        if not start: # all K^M histories
            hists = [prod for prod in itertools.product(range(K), repeat=m)]
            HISTCACHE[K, m, start] = hists
        else: # all histories available ending at position m
            mhists = [tuple([K]*(M-m))] # by convention
            if m > 0:
                rems = get_hists(K, m)
                first = mhists[0]
                mhists = [first + rem for rem in rems]
            HISTCACHE[K, m, start] = mhists
    return HISTCACHE[K, m, start]


def batch_var_elim(x, ems, trans):
    """
    x - T x bsz
    trans - K+1 x K+1 x ... x K log normalized transition mat
    emds - K x V log normalized
    returns bsz-length tensor
    """
    T, K = len(x), ems.size(0)
    M = trans.dim() - 1
    # make possible histories for each position
    fullhists = get_hists(K, M)
    allhists = []
    for m in range(M):
        allhists.append(get_hists(K, m, M=M, start=True))
    # have full history for everyone else
    [allhists.append(fullhists) for _ in range(M, T)]
    # initialize base case
    #tau = {hist: 0 for hist in allhists[T-1]}
    tau = defaultdict(float) # defaults to 0
    for t in range(T-1, -1, -1):
        nutau = {}
        for hist in allhists[t]: # sequence of length M
            terms = [_multi_idx_loop(trans, hist)[k] + ems[k][x[t]]
                     + tau[hist[1:] + (k,)] for k in range(K)]
            # stack terms to get K x bsz tensor
            nutau[hist] = torch.logsumexp(torch.stack(terms), dim=0)
        tau = nutau
    assert len(tau) == 1
    return tau[allhists[0][0]]


def batch_fwdalg_justfo_rul(x, ems, trans, save=False):
    """
    x - T x bsz
    trans - K+1 x K log normalized transition mat
    ems - K x V log normalized
    returns bsz-length tensor
    """
    T, bsz = x.size()
    K = ems.size(0)
    M = trans.dim() - 1
    assert M == 1
    em_0 = ems.t()[x[0]] # bsz x K
    alph = em_0 + trans[K].view(1, K) # bsz x K
    if save:
        table = [alph]
    for t in range(1, T):
        em_t = ems.t()[x[t]] # bsz x K
        # sum over all transitions to get bsz x K
        nualph = torch.logsumexp(alph.unsqueeze(2).expand(bsz, K, K)
                                 + trans[:K].unsqueeze(0).expand(bsz, K, K), dim=1)
        alph = nualph + em_t # bsz x K
        if save:
            table.append(alph)
    if save:
        return torch.logsumexp(alph, dim=1), table
    else:
        return torch.logsumexp(alph, dim=1)


def batch_fwdalg_justfo(x, ems, trans, save=False):
    """
    x - T x bsz
    trans - bsz x K+1 x K log normalized transition mat
    ems - bsz x K x V log normalized
    returns bsz-length tensor
    """
    T, bsz = x.size()
    K = ems.size(1)
    M = trans.dim() - 2
    assert M == 1
    #em_0 = ems.t()[x[0]] # bsz x K
    em_0 = ems.transpose(1, 2).gather(1, x[0].view(bsz, 1, 1).expand(bsz, 1, K)) # bsz x 1 x K
    alph = em_0.squeeze() + trans[:, K] # bsz x K
    if save:
        table = [alph]
    for t in range(1, T):
        em_t = ems.transpose(1, 2).gather(1, x[t].view(bsz, 1, 1).expand(bsz, 1, K)) # bsz x 1 x K
        # sum over all transitions to get bsz x K
        nualph = torch.logsumexp(alph.unsqueeze(2).expand(bsz, K, K)
                                 + trans[:, :K], dim=1)
        alph = nualph + em_t.squeeze() # bsz x K
        if save:
            table.append(alph)
    if save:
        return torch.logsumexp(alph, dim=1), table
    else:
        return torch.logsumexp(alph, dim=1)


# see https://ayorho.files.wordpress.com/2011/05/chapter7.pdf
# and http://infohost.nmt.edu/~olegm/489/Scott2002HMM.pdf for how to do this.
# the idea is we factorize the posterior as p(z_t|x, z_{t+1:T}). we have:
# p(z_t|x, z_{t+1:T}) \prop p(z_{t:T}, x) = alph[t]*p(z_{t+1}|z_t)*p(z_{t+2:T}, x_{t+1:T}|z_{t+1})
#                                         \prop alph[t]*p(z_{t+1}|z_t)
# so we can compute the alphas w/ the fwd alg, then go backward from T, sampling from base case
# alph[T], and using sampled value at t+1 to sample t
def batch_posterior_sample(x, ems, trans, nsamps=1):
    """
    x - T x bsz
    trans - bsz x K+1 x K log normalized transition mat
    ems - bsz x K x V log normalized
    returns T x bsz*nsamps samples and bsz*nsamps posterior log probs
    """
    T, bsz = x.size()
    K = ems.size(1)
    lmarg, alph = batch_fwdalg_justfo(x, ems, trans, save=True)
    assert len(alph) == T
    # start with base case: sample p(z_T | x_{1:T}). Note alph[T-1] = ln p(x_{1:T}, z_T)
    q_T = alph[T-1] - torch.logsumexp(alph[T-1], dim=1, keepdim=True) # bsz x K
    z_T = torch.multinomial(q_T.detach().exp(), num_samples=nsamps, replacement=True) # bsz x nsamps
    ln_qz = q_T.gather(1, z_T).view(-1) # bsz*nsamps
    zsamps = [None]*T
    zsamps[-1] = z_T.view(-1) # bsz*nsamps
    for t in range(T-2, -1, -1):
        # form a probability for each transition
        z_tp1 = zsamps[t+1] # bsz*nsamps
        ln_qt = alph[t].unsqueeze(2) + trans[:, :K] # bsz x K x K, only propto
        # normalize over z_t, conditioned on choice of z_{t+1}
        ztp1_idxs = z_tp1.view(bsz, nsamps).unsqueeze(1).expand(bsz, K, nsamps)
        # amazingly this seems to select columns corresponding to multiple z_{t+1}s
        scores = ln_qt.gather(2, ztp1_idxs) # bsz x K x nsamps;
        # normalize etc so we can get a sample: bsz x nsamps x K
        q_t = scores.transpose(1, 2) - torch.logsumexp(scores.transpose(1, 2), dim=2, keepdim=True)
        z_t = torch.multinomial(q_t.detach().exp().view(bsz*nsamps, K),  # bsz*nsamps x 1
                                num_samples=1)
        zsamps[t] = z_t.view(-1) # bsz*nsamps
        q_t = q_t.contiguous()
        ln_qz = ln_qz + q_t.view(bsz*nsamps, K).gather(1, z_t).view(-1) # bsz*nsamps

    return torch.stack(zsamps), ln_qz


# def batch_posterior_sample_wtf(x, ems, trans, nsamps=1):
#     """
#     x - T x bsz
#     trans - bsz x K+1 x K log normalized transition mat
#     ems - bsz x K x V log normalized
#     returns T x bsz*nsamps samples and bsz*nsamps posterior log probs
#     """
#     T, bsz = x.size()
#     K = ems.size(1)
#     lmarg, alph = batch_fwdalg_justfo(x, ems, trans, save=True)
#     assert len(alph) == T
#     # start with base case: sample p(z_T | x_{1:T}). Note alph[T-1] = ln p(x_{1:T}, z_T)
#     temp = 1000
#     # q_T = (alph[T-1]*temp) - torch.logsumexp(alph[T-1]*temp, dim=1, keepdim=True) # bsz x K
#     q_T = alph[T-1] - torch.logsumexp(alph[T-1], dim=1, keepdim=True) # bsz x K
#     # z_T = torch.multinomial(q_T.detach().exp(), num_samples=nsamps, replacement=True) # bsz x nsamps
#     _, z_T = q_T.max(dim=1, keepdim=True)
#     z_T = z_T.repeat(1, nsamps)
#     ln_qz = q_T.gather(1, z_T).view(-1) # bsz*nsamps
#     zsamps = [None]*T
#     zsamps[-1] = z_T.view(-1) # bsz*nsamps
#     for t in range(T-2, -1, -1):
#         # form a probability for each transition
#         z_tp1 = zsamps[t+1] # bsz*nsamps
#         ln_qt = alph[t].unsqueeze(2) + trans[:, :K] # bsz x K x K, only propto
#         # normalize over z_t, conditioned on choice of z_{t+1}
#         ztp1_idxs = z_tp1.view(bsz, nsamps).unsqueeze(1).expand(bsz, K, nsamps)
#         # amazingly this seems to select columns corresponding to multiple z_{t+1}s
#         scores = ln_qt.gather(2, ztp1_idxs) # bsz x K x nsamps;
#         #scores = temp*scores
#         # normalize etc so we can get a sample: bsz x nsamps x K
#         q_t = scores.transpose(1, 2) - torch.logsumexp(scores.transpose(1, 2), dim=2, keepdim=True)
#         # z_t = torch.multinomial(q_t.detach().exp().view(bsz*nsamps, K),  # bsz*nsamps x 1
#         #                         num_samples=1, replacement=True)
#         _, z_t = q_t.contiguous().view(bsz*nsamps, K).max(dim=1, keepdim=True)
#         zsamps[t] = z_t.view(-1) # bsz*nsamps
#         q_t = q_t.contiguous()
#         ln_qz = ln_qz + q_t.view(bsz*nsamps, K).gather(1, z_t).view(-1) # bsz*nsamps
#
#     return torch.stack(zsamps), ln_qz



def _get_startlps(trans, K, M):
    if M == 1:
        slps = trans[K]
    elif M == 2:
        slps = trans[K, K]
    elif M == 3:
        slps = trans[K, K, K]
    elif M == 4:
        slps = trans[K, K, K, K]
    return slps

def _wo_startlps(trans, K, M):
    if M == 1:
        tlps = trans[:K]
    elif M == 2:
        tlps = trans[:K, :K]
    elif M == 3:
        tlps = trans[:K, :K, :K]
    elif M == 4:
        tlps = trans[:K, :K, :K, :K]
    return tlps

def _em_expand(em, M):
    bsz, K = em.size()
    if M == 0:
        emex = em
    elif M == 1:
        emex = em.view(bsz, 1, K)
    elif M == 2:
        emex = em.view(bsz, 1, 1, K)
    elif M == 3:
        emex = em.view(bsz, 1, 1, 1, K)
    return emex

def batch_fwdalg(x, ems, trans):
    """
    x - T x bsz
    trans - K+1 x K+1 x ... x K log normalized transition mat
    ems - K x V log normalized
    returns bsz-length tensor
    """
    T, bsz = x.size()
    K = ems.size(0)
    M = trans.dim() - 1
    assert M <= 4

    tlps0 = _get_startlps(trans, K, M)
    em_0 = ems.t()[x[0]] # bsz x K
    alph = em_0 + tlps0.view(1, K) # bsz x K

    for m in range(1, min(M, T)):
        slpsm = _get_startlps(trans, K, M-m) # K+1^m x K
        tlpsm = _wo_startlps(slpsm, K, m) # K^m x K
        # use broadcasting to get something bsz x K^{m+1}
        nualph = alph.unsqueeze(m+1) + tlpsm.unsqueeze(0)
        em_m = ems.t()[x[m]]
        alph = nualph + _em_expand(em_m, m)

    justrans = _wo_startlps(trans, K, M) # K^{M+1}

    for t in range(M, T):
        # sum over all first-dim transitions to get bsz x K^M
        nualph = torch.logsumexp(alph.unsqueeze(M+1)
                                 + justrans.unsqueeze(0), dim=1)
        em_t = ems.t()[x[t]] # bsz x K
        alph = (nualph + _em_expand(em_t, M-1))

    return torch.logsumexp(alph.view(bsz, -1), dim=1)


def _pw_expand(pot, m, K):
    """
    pot is K*K
    """
    if m == 1:
        expot = pot.view(K, K)
    elif m == 2:
        expot = pot.view(K, 1, K)
    elif m == 3:
        expot = pot.view(K, 1, 1, K)
    elif m == 4:
        expot = pot.view(K, 1, 1, 1, K)
    return expot

# DP for M = 3 looks like:
# alph_t[j, k, l] = ln p(x_t | z_t=l) ln psi_2(j, l) + ln psi_1(k, l)
#                   + ln \sum_i exp ( alph_{t-1}[i, j, k] + ln psi_3(i, l) )
# for tied factors can actually do a lot of this outside the loop...
def batch_ufwdalg(pwlpots, T, K, M, edges, ulpots=None, ending_at=None, save=False):
    """
    pwlpots - nedges x K*K
    T, M - length, markov order
    edges - nedges x 2
    ulpots - bsz x T x K
    """
    if ulpots is None:
        bsz = 1
        ulpots = pwlpots.new(1, K).zero_().view(1, 1, K).expand(T, 1, K)
    else:
        bsz = ulpots.size(0)
        ulpots = ulpots.transpose(0, 1).contiguous() # T x bsz x K

    # just for now: maps each t to list of edge_indices ending at t in ascending order
    # of start vertex; assumes edges are in our canonical order
    if ending_at is None:
        ending_at = {t: [] for t in range(1, T)}
        for n in range(pwlpots.size(0)):
            start, end = edges[n]
            start, end = start.item(), end.item()
            ending_at[end].append(n)

    alph = ulpots[0] # bsz x K, no start probs for now
    if save:
        table = [alph.detach().clone()]

    # we don't actually have to sum over transitions for first M things
    for m in range(1, min(M, T)): # handle fewer links etc
        facidxs = ending_at[m]
        rem = pwlpots[facidxs[-1]].view(K, K)
        for r in range(2, min(m+1, M)): # form K^M tensor of potentials
            rem = _pw_expand(pwlpots[facidxs[-r]], r, K) + rem.unsqueeze(0)
        alph = alph.unsqueeze(m+1) + rem.unsqueeze(0) + _em_expand(ulpots[m], m)
        if save:
            table.append(alph.detach().clone())

    for t in range(M, T):
        facidxs = ending_at[t] # list of length M
        # first sum over all M-transitions
        nualph = torch.logsumexp(alph.unsqueeze(M+1)
                                 + _pw_expand(pwlpots[facidxs[0]], M, K).unsqueeze(0), dim=1)

        # now add remaining potentials, in reverse order for convenience
        if M > 1:
            rem = pwlpots[facidxs[-1]].view(K, K)
            for r in range(2, M): # form K^M tensor of potentials
                rem = _pw_expand(pwlpots[facidxs[-r]], r, K) + rem.unsqueeze(0)
            nualph = nualph + rem.unsqueeze(0)

        alph = nualph + _em_expand(ulpots[t], M-1)
        if save:
            table.append(alph.detach().clone())

    if save:
        return torch.logsumexp(alph.view(bsz, -1), dim=1), table
    else:
        return torch.logsumexp(alph.view(bsz, -1), dim=1)


def batch_ubwdalg(pwlpots, T, K, M, edges, ulpots=None, ending_at=None, save=False):
    """
    pwlpots - nedges x K*K
    T, M - length, markov order
    edges - nedges x 2
    ulpots - bsz x T x K
    """
    if ulpots is None:
        ulpots = pwlpots.new(1, K).zero_().view(1, 1, K).expand(T, 1, K)
    else:
        ulpots = ulpots.transpose(0, 1).contiguous() # T x bsz x K

    table = [None] # just zeros

    # since beta[T] = 0, do T-1 separately
    if T > M:
        facidxs = ending_at[T-2+1]
        trans = pwlpots[facidxs[M-1]].view(K, K)
        for r in range(M-2, -1, -1):
            trans = _pw_expand(pwlpots[facidxs[r]], M-r, K) + trans.unsqueeze(0)
        beta = torch.logsumexp(trans.unsqueeze(0) + _em_expand(ulpots[T-2+1], M), dim=M+1)
        if save:
            table.append(beta.detach().clone())
    else:
        beta = 0

    # timesteps T-2 thru M
    for t in range(T-3, M-2, -1):
        facidxs = ending_at[t+1]
        trans = pwlpots[facidxs[M-1]].view(K, K)
        for r in range(M-2, -1, -1):
            trans = _pw_expand(pwlpots[facidxs[r]], M-r, K) + trans.unsqueeze(0)
        beta = torch.logsumexp(beta.unsqueeze(1) + trans.unsqueeze(0)
                               + _em_expand(ulpots[t+1], M), dim=M+1)
        if save:
            table.append(beta.detach().clone())

    # timesteps M-1 thru 1
    for m in range(M-2, -1, -1):
        facidxs = ending_at[m+1]
        trans = pwlpots[facidxs[m]].view(K, K)
        for r in range(m-1, -1, -1):
            trans = _pw_expand(pwlpots[facidxs[r]], m-r+1, K) + trans.unsqueeze(0)
        beta = torch.logsumexp(beta + trans.unsqueeze(0)
                               + _em_expand(ulpots[m+1], m+1), dim=m+2)
        if save:
            table.append(beta.detach().clone())

    # final step
    beta = torch.logsumexp(beta + ulpots[0], dim=1)
    if save:
        return beta, table[::-1], ulpots
    else:
        return beta


# just hardcoding for now
def _do_rest_logmarg(table, M):
    """
    table is bsz x K^{M+1}, returns marginals in descending order of end-edge
    """
    if M == 2:
        margs = [table.logsumexp(3)]
    elif M == 3:
        margs = [table.logsumexp(1).logsumexp(3), table.logsumexp(2).logsumexp(3),
                 table.logsumexp(3).logsumexp(3)]
    elif M == 4:
        margs = [table.logsumexp(1).logsumexp(1).logsumexp(3), table.logsumexp(1).logsumexp(2).logsumexp(3),
                 table.logsumexp(2).logsumexp(2).logsumexp(3), table.logsumexp(1).logsumexp(3).logsumexp(3),
                 table.logsumexp(2).logsumexp(3).logsumexp(3), table.logsumexp(3).logsumexp(3).logsumexp(3)]
    return margs


# just hardcoding for now
def _do_logmarg(table, M):
    """
    table is bsz x K^{M+1}, returns marginals in descending order of end-edge
    """
    if M == 1:
        margs = [table.clone()]
    elif M == 2:
        margs = [table.logsumexp(1), table.logsumexp(2)]
        #margs = [table.logsumexp(2), table.logsumexp(1)]
    elif M == 3:
        margs = [table.logsumexp(1).logsumexp(1), table.logsumexp(1).logsumexp(2),
                 table.logsumexp(2).logsumexp(2)]
        # margs = [table.logsumexp(2).logsumexp(2), table.logsumexp(1).logsumexp(2),
        #          table.logsumexp(1).logsumexp(1)]
    elif M == 4:
        margs = [table.logsumexp(1).logsumexp(1).logsumexp(1), table.logsumexp(1).logsumexp(1).logsumexp(2),
                 table.logsumexp(1).logsumexp(2).logsumexp(2), table.logsumexp(2).logsumexp(2).logsumexp(2)]
        # margs = [table.logsumexp(2).logsumexp(2).logsumexp(2), table.logsumexp(1).logsumexp(2).logsumexp(2),
        #          table.logsumexp(1).logsumexp(1).logsumexp(2), table.logsumexp(1).logsumexp(1).logsumexp(1)]
    return margs

def exact_marginals(pwlpots, T, K, M, edges, ulpots=None, ending_at=None):
    flmarg, ftable = batch_ufwdalg(pwlpots, T, K, M, edges, ulpots=ulpots,
                                   ending_at=ending_at, save=True)
    blmarg, btable, ulpots = batch_ubwdalg(pwlpots, T, K, M, edges, ulpots=ulpots,
                                           ending_at=ending_at, save=True)

    # note marginals on clique_size-1 chunks are given by ftable[t] + btable[t]

    edge_lmargs = [None] * edges.size(0)
    node_lmargs = [None] * T

    if T > M:
        # do last M+1 chunk first, since it doesn't need Beta
        facidxs = ending_at[T-1]
        trans = pwlpots[facidxs[M-1]].view(K, K)
        for r in range(M-2, -1, -1):
            trans = _pw_expand(pwlpots[facidxs[r]], M-r, K) + trans.unsqueeze(0)
        clmarg_t = ftable[T-2].unsqueeze(M+1) + trans.unsqueeze(0) + _em_expand(ulpots[T-1], M)
        # now calculate edge marginals from the clique marginal
        elmargs = _do_logmarg(clmarg_t, M)
        for r in range(M):
            edge_lmargs[facidxs[M-r-1]] = elmargs[r]
        # calculate node marginal from any of these
        node_lmargs[T-1] = elmargs[0].logsumexp(1)

        for t in range(T-2, M-1, -1):
            facidxs = ending_at[t]
            trans = pwlpots[facidxs[M-1]].view(K, K)
            for r in range(M-2, -1, -1):
                trans = _pw_expand(pwlpots[facidxs[r]], M-r, K) + trans.unsqueeze(0)
            clmarg_t = (ftable[t-1].unsqueeze(M+1) + trans.unsqueeze(0)
                        + _em_expand(ulpots[t], M) + btable[t].unsqueeze(1))
            elmargs = _do_logmarg(clmarg_t, M)
            for r in range(M):
                edge_lmargs[facidxs[M-r-1]] = elmargs[r]
            node_lmargs[t] = elmargs[0].logsumexp(1)

        if M > 1: # need to handle edges ending before M+1
            restmargs = _do_rest_logmarg(clmarg_t, M) # this is the clmarg_t from t=M
            restidx = 0
            for m in range(M-1, 0, -1):
                facidxs = ending_at[m]
                node_lmargs[m] = restmargs[restidx].logsumexp(1)
                for r in range(len(facidxs)):
                    edge_lmargs[facidxs[len(facidxs)-r-1]] = restmargs[restidx]
                    restidx += 1
            # marginal of node zero
            node_lmargs[0] = restmargs[-1].logsumexp(2)
        else:
            node_lmargs[0] = elmargs[-1].logsumexp(2)
    else: # we have 1 < T <= M
        btab = btable[T-1] if btable[T-1] is not None else 0
        restmargs = _do_logmarg(ftable[T-1] + btab, T-1)
        restidx = 0
        for m in range(T-1, 0, -1):
            facidxs = ending_at[m]
            node_lmargs[m] = restmargs[restidx].logsumexp(1)
            for r in range(len(facidxs)):
                edge_lmargs[facidxs[len(facidxs)-r-1]] = restmargs[restidx]
                restidx += 1
        node_lmargs[0] = restmargs[-1].logsumexp(2)

    everybody = [nlmrg for nlmrg in node_lmargs]
    everybody.extend([elmrg.view(-1, K*K) for elmrg in edge_lmargs])
    allmargs = torch.exp(torch.cat(everybody, 1) - flmarg.view(-1, 1))

    return allmargs, edge_lmargs, node_lmargs, flmarg, ftable, btable
