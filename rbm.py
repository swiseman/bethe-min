import os
import math
import argparse
import itertools
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.functional import softplus

from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

#from rbm_infnets import RBMInfConvNet, RBMRNNProdInfNet, RBMConvProdInfNet
from rbm_infnets import SeqInfNet, SeqJustVisInfNet, DblSeqInfNet, TwodJustVisInfNet

from utils import clip_opt_params, get_rbm_ne, batch_kl, get_rbm_edges
from lbp_util import dolbp
#import kuleshov_data_py3 as data


class RBM(nn.Module):
    def __init__(self, nvis, opt):
        super(RBM, self).__init__()
        self.nvis, self.nhid = nvis, opt.nhid
        # self.drop = nn.Dropout(opt.dropout)
        self.W = nn.Parameter(torch.Tensor(self.nvis, opt.nhid))
        self.b = nn.Parameter(torch.Tensor(self.nvis, 1))
        self.a = nn.Parameter(torch.Tensor(1, opt.nhid))
        self.init_strat = opt.init_strat
        self.pinit = opt.init
        self.last_h_sample = None
        self.init_weights()

    def init_weights(self):
        # everyone seems to init biases at zero
        self.b.data.zero_()
        self.a.data.zero_()
        if self.init_strat == "xavier":
            u = 4 * math.sqrt(6.0 / (self.nvis + self.nhid))
            self.W.data.uniform_(-u, u)
        elif self.init_strat == "gaussian":
            self.W.data.normal_(0, self.pinit)
        else:
            self.W.data.uniform_(-self.pinit, self.pinit)
            self.a.data.uniform_(-self.pinit, self.pinit)
            self.b.data.uniform_(-self.pinit, self.pinit)

    # some standard rbm stuff
    def _neg_energy(self, v, h):
        """
        v - bsz x nvis
        h - bsz x nhid
        returns bsz vector corresponding to v' W h + v'b + h'a
        """
        return (v.mm(self.b.view(-1, 1)).squeeze() + h.mm(self.a.view(-1, 1)).squeeze()
                + (h.mm(self.W.t())*v).sum(1))

    def _neg_free_energy(self, v):
        """
        returns bsz vector corresponding to log sum_h exp(v' W h + v'b + h'a)
        """
        return v.mm(self.b.view(-1, 1)).squeeze() + softplus(v.mm(self.W) + self.a).sum(1)

    def _logmarg_v(self, h):
        """
        returns bsz vector corresponding to log sum_v exp(v' W h + v'b + h'a)
        """
        return h.mm(self.a.view(-1, 1)).squeeze() + softplus(h.mm(self.W.t()) + self.b.t()).sum(1)

    def _brute_marginalize(self):
        nvis, nhid = self.nvis, self.nhid
        if nhid > nvis:
            all_vs = self.W.new(list(itertools.product([0, 1], repeat=nvis))) # 2^nvis x nvis
            nfes = self._neg_free_energy(all_vs)
        else:
            all_hs = self.W.new(list(itertools.product([0, 1], repeat=nhid))) # 2^nhid x nhid
            nfes = self._logmarg_v(all_hs)
        return torch.logsumexp(nfes, dim=0)

    def _sample_hiddens(self, v):
        """
        returns bsz x nhid samples
        """
        return torch.bernoulli(torch.sigmoid(v.mm(self.W) + self.a))

    def _sample_visibles(self, h):
        """
        returns bsz x nvis samples
        """
        return torch.bernoulli(torch.sigmoid(h.mm(self.W.t()) + self.b.t()))

    def rb_pcd_loss(self, v):
        """
        v - bsz x nvis
        returns bsz loss vector
        """
        # get exact gradient of first term; equivalent to rao blackwelized etc
        loss = -self._neg_free_energy(v).sum()
        with torch.no_grad():
            if self.last_h_sample is None: # sklearn initializes from zero
                self.last_h_sample = v.new(v.size(0), self.nhid).zero_()
            v_squig = self._sample_visibles(self.last_h_sample)
            # use mean h-vector rather than sample (for rao-blackwelization)
            h_squiq_mean = torch.sigmoid(v_squig.mm(self.W) + self.a)
            self.last_h_sample = torch.bernoulli(h_squiq_mean)
        # approximate partition function with these samples
        loss = loss + self._neg_energy(v_squig, h_squiq_mean).sum()
        return loss

    def _random_pseudo_ll(self, v):
        """
        v - bsz x nvis
        returns bsz-array
        """
        vflip = v.detach().clone()
        ridxs = torch.LongTensor(v.size(0)).random_(0, v.size(1)).to(v.device)
        orig = v.gather(1, ridxs.view(-1, 1))
        # flip
        vflip.scatter_(1, ridxs.view(-1, 1), (1-orig))
        nfe = self._neg_free_energy(v) # bsz
        fnfe = self._neg_free_energy(vflip) # bsz
        return v.size(1)*nn.functional.logsigmoid(nfe - fnfe)

    def get_edge_scores(self):
        """
        returns nvis x nhid x 2*2 log potentials; we'll move unaries into pairwise
        """
        nvis, nhid = self.W.size()
        # divide biases between hidden or visible so we can just use pw potentials
        W = self.W
        first3 = torch.zeros_like(W).unsqueeze(2) # nvis x nhid x 1
        lpots = torch.cat([first3.expand(nvis, nhid, 3), W.unsqueeze(2)], 2)
        return lpots

    def get_unary_scores(self):
        """
        returns (nvis + nhid) x 2 lpots
        """
        vislpots = torch.cat([torch.zeros_like(self.b), self.b], 1) # nvis x 2
        hidlpots = torch.cat([torch.zeros_like(self.a).t(), self.a.t()], 1) # nhid x 2
        # concat
        return torch.cat([vislpots, hidlpots], 0)


# see https://www.cs.toronto.edu/~rsalakhu/papers/dbn_ais.pdf
class RBMAIS(object):
    def __init__(self, rbm, b_A, nhid_A):
        self.rbm = rbm
        self.nhid_A = nhid_A
        self.b_A = b_A
        self.logZ_A = self.nhid_A*math.log(2) + softplus(b_A.view(-1)).sum()

    def init_sample(self):
        return torch.bernoulli(torch.sigmoid(self.b_A.view(-1)))

    def T_k(self, v, beta_k):
        """
        v - 1 x nvis
        returns 1 x nvis
        """
        # (dont need to sample h's from base; should always be 0.5 since no params for hiddens)
        # sample h's from rbm given v
        h_B = torch.bernoulli(torch.sigmoid(beta_k*(v.mm(self.rbm.W) + self.rbm.a))) # 1 x nhid
        # sample new v
        vnu = torch.bernoulli(torch.sigmoid( # 1 x nvis
            (1-beta_k)*self.b_A.view(1, -1)
            + beta_k*(h_B.mm(self.rbm.W.t()) + self.rbm.b.t())))
        return vnu

    def log_ptilde_k(self, v, beta_k):
        """
        returns 0 dim log ptilde_k, where ptilde_k is propto p_k(v)
        """
        alnterm = (1-beta_k)*v.view(-1).dot(self.b_A.view(-1))
        blnterm = (beta_k*v.mv(self.rbm.b.view(-1))
                   + softplus(beta_k*(v.mm(self.rbm.W) + self.rbm.a)).sum(1))
        return alnterm + blnterm.squeeze()

    def run_ais(self, M, nsteps=1000):
        """
        M - number of chains or whatever
        """
        steps = torch.linspace(0, 1, steps=nsteps)
        lograt_estimates = []
        for _ in range(M):
            # sample initial
            v1 = self.init_sample().view(1, -1) # 1 x nvis
            prev_lptilde = self.log_ptilde_k(v1, 0).item()
            w_m = -prev_lptilde
            v_k = v1
            for k in range(1, steps.size(0)):
                beta_k = steps[k].item()
                w_m += self.log_ptilde_k(v_k, beta_k).item()
                if k < steps.size(0) - 1:
                    v_k = self.T_k(v_k, beta_k)
                    w_m -= self.log_ptilde_k(v_k, beta_k).item()
            lograt_estimates.append(w_m)
        # estimate of ratio is
        estrat = torch.logsumexp(torch.tensor(lograt_estimates), dim=0) - math.log(M)
        logZ_B_est = estrat + self.logZ_A
        return logZ_B_est, estrat

# this assumes unary potentials, as in the standard rbm treatment
def rbm_bethe_fez2(tau_u, tau_e, un_lpots, ed_lpots, ne):
    """
    tau_u - 1 x (nvis + nhid) x 2
    tau_e - 1 x nvis x nhid x 2*2
    un_lpots - 1 x (nvis + nhid) x 2
    ed_lpots - 1 x nvis x nhid x 2*2
    ne - 1 x (nvis + nhid) x 2
    """
    assert tau_e.size(0) == ed_lpots.size(0)
    assert tau_u.size(0) == un_lpots.size(0)
    en = -(tau_u * un_lpots).sum() - (tau_e * ed_lpots).sum()
    negfacent = (tau_u * tau_u.log()).sum() + (tau_e * tau_e.log()).sum()
    nodent = (ne * tau_u * tau_u.log()).sum()
    return en + negfacent - nodent


def get_pred_lmargs(pw_logits, V, H, geom=False):
    """
    pw_logits - V*H x K^2 pairwise factor logits; assume each pairwse factor is V var by H var
    returns:
      - V*H x K^2 log marginals for each configuration
      - V*H x K   predicted row log marginals (i.e., predicted v log marginals for each h)
      - V*H x K   predicted col log marginals (i.e., predicted h log marginals for each v)
      - V x K     implied row/v log marginals (obtained by averaging)
      - H x K     implied col/h log marginals (obtained by averaging)
    """
    K = int(math.sqrt(pw_logits.size(1)))
    # constraints are either of the form:
    # LSE(col_or_row_logits_i) - ln Z_k = LSE(col_or_row_logits_j) - ln Z_l
    # or:
    # LSE(col_or_row_logprobs_i) = log marginal_j
    all_pw_logZs = pw_logits.logsumexp(1).view(-1, 1) # V*H x 1
    pred_rowlmargs = pw_logits.view(-1, K, K).logsumexp(2) # V*H x K (un log normalized)
    pred_collmargs = pw_logits.view(-1, K, K).logsumexp(1) # V*H x K (un log normalized)
    # log normalize so we get row and column log marginals
    pred_rowlmargs = pred_rowlmargs - all_pw_logZs
    pred_collmargs = pred_collmargs - all_pw_logZs
    # get V+H implied log margs by avging row/col log margs either geometrically or arithmetically
    if geom: # geometric avg of unlogged row/col margs
        vlmargs = pred_rowlmargs.view(V, H, -1).mean(1) # V x K log marginals
        hlmargs = pred_collmargs.view(V, H, -1).mean(0) # H x K log marginals
    else: # probably a better idea
        vlmargs = pred_rowlmargs.view(V, H, -1).logsumexp(1) - math.log(H) # V x K log marginals
        hlmargs = pred_collmargs.view(V, H, -1).logsumexp(0) - math.log(V) # H x K log marginals
    return (pw_logits - all_pw_logZs), pred_rowlmargs, pred_collmargs, vlmargs, hlmargs


def get_lmarg_penalties(pred_rowlmargs, pred_collmargs, vlmargs, hlmargs, penfunc):
    """
    pred row and col lmargs are V*H x K
    vlmargs - V x K
    hlmargs - H x K
    returns V x H tensor of penalties
    """
    # constraints are either of the form:
    # LSE(col_or_row_logits_i) - ln Z_k = LSE(col_or_row_logits_j) - ln Z_l
    # or:
    # LSE(col_or_row_logprobs_i) = log marginal_j
    V, H = vlmargs.size(0), hlmargs.size(0)
    vpen = penfunc(pred_rowlmargs.view(V, H, -1), vlmargs.view(V, 1, -1).expand(V, H, -1))
    hpen = penfunc(pred_collmargs.view(V, H, -1), hlmargs.view(1, H, -1).expand(V, H, -1))
    return vpen, hpen


def get_taus_and_pens(rho, V, H, penfunc=None):
    (pwlmargs, pred_vlmargs, pred_hlmargs,
     av_vlmargs, av_hlmargs) = get_pred_lmargs(rho, V, H)
    if penfunc is not None:
        vpen, hpen = get_lmarg_penalties(pred_vlmargs, pred_hlmargs, av_vlmargs,
                                         av_hlmargs, penfunc)
        penloss = vpen.sum() + hpen.sum()
    else:
        penloss = 0
    tau_u = torch.cat([av_vlmargs, av_hlmargs], 0).unsqueeze(0) # 1 x nvis+nhid x 2
    tau_e = pwlmargs.unsqueeze(0) # 1 x V*H x K^2
    return tau_u, tau_e, penloss

EPS = 1e-38

def rbm_outer_loss(batch, model, rho, un_lpots, ed_lpots, ne, penfunc=None):
    """
    batch - bsz x nvis
    un_lpots - 1 x nvis + nhid x 2
    ed_lpots - 1 x nvis x nhid x 2*2
    ne - 1 x nvis + nhid x 2
    """
    # loss is -log \sum_h exp(-en(h, v)) - F(tau, theta)
    #       = free_energy(v) - F(tau, theta)
    _, V, H, _ = ed_lpots.size()
    bsz = batch.size(0)
    tau_u, tau_e, penloss = get_taus_and_pens(rho, V, H, penfunc=penfunc)
    tau_u, tau_e = tau_u.exp() + EPS, tau_e.exp() + EPS
    fz = rbm_bethe_fez2(tau_u, tau_e.view(1, V, H, -1), un_lpots, ed_lpots, ne)
    # first term just marginalizes out the hs
    nfe = model._neg_free_energy(batch) # bsz
    return -nfe.sum() - fz*bsz, penloss


def rbm_inner_loss(rho, un_lpots, ed_lpots, ne, penfunc):
    """
    rho - V*H x K^2 logits
    un_lpots - 1 x nvis + nhid x 2
    ed_lpots - 1 x nvis x nhid x 2*2
    ne - 1 x nvis + nhid x 2
    """
    _, V, H, _ = ed_lpots.size()
    tau_u, tau_e, penloss = get_taus_and_pens(rho, V, H, penfunc=penfunc)
    tau_u, tau_e = tau_u.exp() + EPS, tau_e.exp() + EPS
    fz = rbm_bethe_fez2(tau_u, tau_e.view(1, V, H, -1), un_lpots, ed_lpots, ne)
    return fz, penloss


def train(corpus, model, infnet, moptim, ioptim, ne, penfunc, device, args):
    """
    opt is just gonna be for everyone
    ne - 1 x nvis*nhid
    """
    #import time
    model.train()
    infnet.train()
    total_out_loss, total_in_loss, nexamples = 0.0, 0.0, 0
    total_pen_loss = 0.0

    for i, batchthing in enumerate(corpus):
        batch = batchthing[0] # i guess it's a list
        batch = batch.view(batch.size(0), -1).to(device) # bsz x V
        bsz, nvis = batch.size()
        npenterms = 2*nvis*model.nhid # V for each H marginal and H for each V marginal

        # maximize wrt rho
        with torch.no_grad():
            ed_lpots = model.get_edge_scores().unsqueeze(0) # 1 x nvis x nhid x 4
            un_lpots = model.get_unary_scores().unsqueeze(0) # 1 x (nvis + nhid) x 2

        if args.reset_adam:
            ioptim = torch.optim.Adam(infnet.parameters(), lr=args.ilr)

        for _ in range(args.inf_iter):
            ioptim.zero_grad()
            pred_rho = infnet.q() # V*H x K^2 logits
            in_loss, pen_loss = rbm_inner_loss(pred_rho, un_lpots, ed_lpots, ne, penfunc)
            total_in_loss += in_loss.item()*bsz
            total_pen_loss += args.pen_mult/npenterms * pen_loss.item()*bsz
            in_loss = in_loss + args.pen_mult/npenterms * pen_loss
            in_loss.backward()
            clip_opt_params(ioptim, args.clip)
            ioptim.step()
        pred_rho = pred_rho.detach()

        # min wrt params
        moptim.zero_grad()
        ed_lpots = model.get_edge_scores().unsqueeze(0) # 1 x nvis x nhid x 4
        un_lpots = model.get_unary_scores().unsqueeze(0) # 1 x (nvis + nhid) x 2

        out_loss, open_loss = rbm_outer_loss(batch, model, pred_rho, un_lpots, ed_lpots,
                                             ne, penfunc=None)
        total_out_loss += out_loss.item()
        out_loss.div(bsz).backward()
        clip_opt_params(moptim, args.clip)
        moptim.step()
        nexamples += bsz

        if (i+1) % args.log_interval == 0:
            print("{:5d}/{:5d} | out_loss {:8.5f} | in_loss {:8.5f} | pen_loss {:8.6f}".format(
                i+1, len(corpus), total_out_loss/nexamples, total_in_loss/nexamples,
                total_pen_loss/(nexamples*args.pen_mult)))

    return total_out_loss, total_in_loss, total_pen_loss, nexamples


def lbp_train(corpus, model, optimizer, edges, ne, device, args):
    """
    opt is just gonna be for everyone
    ne - 1 x nvis*nhid
    """
    model.train()
    total_out_loss, nexamples = 0.0, 0
    niter = 0

    for i, batchthing in enumerate(corpus):
        optimizer.zero_grad()
        batch = batchthing[0] # i guess it's a list
        batch = batch.view(batch.size(0), -1).to(device) # bsz x V
        bsz, nvis = batch.size()
        nedges = nvis * model.nhid # V*H edges

        ed_lpots = model.get_edge_scores().unsqueeze(0) # 1 x nvis x nhid x 4
        un_lpots = model.get_unary_scores().unsqueeze(0) # 1 x (nvis + nhid) x 2

        with torch.no_grad():
            exed_lpots = ed_lpots.view(nedges, 1, 2, 2) # V*H x 1 x 2 x 2
            exun_lpots = un_lpots.transpose(0, 1) # V+H x 1 x 2
            nodebs, facbs, ii, _, _ = dolbp(exed_lpots, edges, x=None, emlps=exun_lpots,
                                            niter=args.lbp_iter, renorm=True, tol=args.lbp_tol,
                                            randomize=args.randomize_lbp)
            niter += ii
            # reshape log unary marginals: V+H x 1 x 2 -> 1 x V+H x 2
            tau_u = torch.stack([nodebs[t] for t in range(nvis + model.nhid)]).transpose(0, 1)
            # reshape log fac marginals: nedges x 1 x 2 x 2 -> 1 x nedges x 2 x 2
            tau_e = torch.stack([facbs[e] for e in range(nedges)]).transpose(0, 1)
            # exponentiate
            tau_u, tau_e = (tau_u.exp() + EPS), (tau_e.exp() + EPS).view(1, nvis, model.nhid, -1)

        fz = rbm_bethe_fez2(tau_u, tau_e, un_lpots, ed_lpots, ne)
        out_loss = -model._neg_free_energy(batch).sum() - fz*bsz
        total_out_loss += out_loss.item()
        out_loss.div(bsz).backward()
        clip_opt_params(optimizer, args.clip)
        optimizer.step()
        nexamples += bsz

        if (i+1) % args.log_interval == 0:
            print("{:5d}/{:5d} | its {:3.2f} | out_loss {:8.5f}".format(
                i+1, len(corpus), niter/(i+1), total_out_loss/nexamples))

    return total_out_loss, nexamples


def train_pcd(corpus, model, optimizer, device, args):
    model.train()
    total_loss, nexamples = 0.0, 0

    for i, batchthing in enumerate(corpus):
        batch = batchthing[0] # i guess it's a list
        batch = batch.view(batch.size(0), -1).to(device) # bsz x V
        bsz, nvis = batch.size()
        optimizer.zero_grad()
        loss = model.rb_pcd_loss(batch)
        total_loss += loss.item()
        loss.div(bsz).backward()
        clip_opt_params(optimizer, args.clip)
        optimizer.step()
        nexamples += bsz

        if (i+1) % args.log_interval == 0:
            print("{:5d}/{:5d} | loss {:8.5f}".format(
                i+1, len(corpus), total_loss/nexamples))

    return total_loss, nexamples


def validate(corpus, model, infnet, ne, penfunc, device):
    """
    opt is just gonna be for everyone
    ne - 1 x nvis*nhid
    """
    #import time
    model.eval()
    infnet.eval()
    total_out_loss, total_pen_loss, nexamples = 0.0, 0.0, 0

    for i, batchthing in enumerate(corpus):
        batch = batchthing[0]
        batch = batch.view(batch.size(0), -1).to(device)
        bsz, nvis = batch.size()
        npenterms = 2*nvis*model.nhid

        # maximize wrt rho
        ed_lpots = model.get_edge_scores().unsqueeze(0) # 1 x nvis x nhid x 4
        un_lpots = model.get_unary_scores().unsqueeze(0) # 1 x (nvis + nhid) x 2
        pred_rho = infnet.q() # V*H x K^2 logits

        out_loss, open_loss = rbm_outer_loss(batch, model, pred_rho, un_lpots, ed_lpots,
                                             ne, penfunc=penfunc)
        total_out_loss += out_loss.item()
        total_pen_loss += 1/npenterms * open_loss.item()
        nexamples += bsz

    return total_out_loss, total_pen_loss, nexamples


def lbp_validate(corpus, model, edges, ne, device):
    """
    opt is just gonna be for everyone
    ne - 1 x nvis*nhid
    """
    model.eval()
    total_out_loss, nexamples = 0.0, 0

    for i, batchthing in enumerate(corpus):
        batch = batchthing[0]
        batch = batch.view(batch.size(0), -1).to(device)
        bsz, nvis = batch.size()
        nedges = nvis * model.nhid # V*H edges

        ed_lpots = model.get_edge_scores().unsqueeze(0) # 1 x nvis x nhid x 4
        un_lpots = model.get_unary_scores().unsqueeze(0) # 1 x (nvis + nhid) x 2

        exed_lpots = ed_lpots.view(nedges, 1, 2, 2) # V*H x 1 x 2 x 2
        exun_lpots = un_lpots.transpose(0, 1) # V+H x 1 x 2
        nodebs, facbs, _, _, _ = dolbp(exed_lpots, edges, x=None, emlps=exun_lpots,
                                       niter=args.lbp_iter, renorm=True, tol=args.lbp_tol,
                                       randomize=args.randomize_lbp)

        tau_u = torch.stack([nodebs[t] for t in range(nvis + model.nhid)]).transpose(0, 1)
        tau_e = torch.stack([facbs[e] for e in range(nedges)]).transpose(0, 1)
        tau_u, tau_e = (tau_u.exp() + EPS), (tau_e.exp() + EPS).view(1, nvis, model.nhid, -1)

        fz = rbm_bethe_fez2(tau_u, tau_e, un_lpots, ed_lpots, ne)
        out_loss = -model._neg_free_energy(batch).sum() - fz*bsz
        total_out_loss += out_loss.item()
        nexamples += bsz

    return total_out_loss, nexamples


def moar_validate(corpus, model, device, args, do_ais=False, b_A=None):
    """
    opt is just gonna be for everyone
    ne - 1 x nvis*nhid
    """
    #import time
    model.eval()
    pll, nexamples = 0.0, 0

    if do_ais:
        if b_A is None:
            b_A = model.b.data.clone().uniform_(-0.1, 0.1)
        ais = RBMAIS(model, b_A, model.nhid)

    for i, batchthing in enumerate(corpus):
        batch = batchthing[0]
        batch = batch.view(batch.size(0), -1).to(device)
        bsz, nvis = batch.size()
        if do_ais:
            logZ_B_est, _ = ais.run_ais(args.nais_chains, nsteps=args.nais_steps)
            bll = (model._neg_free_energy(batch) - logZ_B_est).sum()
            pll -= bll
        else:
            bpll = model._random_pseudo_ll(batch).sum().item()
            pll -= bpll
        nexamples += bsz

    return pll, nexamples


parser = argparse.ArgumentParser(description='')

parser.add_argument('-bin_mnist_dir', type=str, default='/scratch/data/binarized_mnist/',
                    help='')
parser.add_argument('-use_kuleshov_data', action='store_true', help='')

parser.add_argument('-nhid', type=int, default=64, help='')
parser.add_argument('-init_strat', type=str, default='xavier',
                    choices=["xavier", "gaussian", "uniform"], help='')

parser.add_argument('-optalg', type=str, default='sgd',
                    choices=["sgd", "adagrad", "adam"], help='')
parser.add_argument('-pen_mult', type=float, default=1, help='')
parser.add_argument('-penfunc', type=str, default="l2",
                    choices=["l2", "l1", "js", "kl1", "kl2"], help='')

parser.add_argument('-infarch', type=str, default="transformer")
                    #choices=["yoon", "rnn", "transformer"], help='')

parser.add_argument('-kW', type=int, default=3, help='')
parser.add_argument('-q_layers', type=int, default=2, help='')
parser.add_argument('-q_heads', type=int, default=4, help='')
parser.add_argument('-qemb_sz', type=int, default=50, help='')
parser.add_argument('-q_hid_size', type=int, default=64, help='')
parser.add_argument('-dropout', type=float, default=0, help='')

parser.add_argument('-init', type=float, default=0.1, help='param init')
parser.add_argument('-qinit', type=float, default=0.1, help='infnet param init')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('-ilr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('-inf_iter', type=int, default=1, help='')
parser.add_argument('-lrdecay', type=float, default=0.5, help='initial learning rate')
parser.add_argument('-pendecay', type=float, default=1, help='initial learning rate')
parser.add_argument('-clip', type=float, default=5, help='gradient clipping')
parser.add_argument('-epochs', type=int, default=40, help='upper epoch limit')
parser.add_argument('-bsz', type=int, default=32, help='batch size')
parser.add_argument('-seed', type=int, default=1111, help='random seed')
parser.add_argument('-cuda', action='store_true', help='use CUDA')

parser.add_argument('-training', type=str, default="bethe",
                    choices=["lbp", "pcd", "bethe"], help='')

parser.add_argument('-lbp_iter', type=int, default=10, help='')
parser.add_argument('-lbp_tol', type=float, default=0.001, help='')
parser.add_argument('-randomize_lbp', action='store_true', help='')
parser.add_argument('-reset_adam', action='store_true', help='')

parser.add_argument('-do_ais', action='store_true', help='')
parser.add_argument('-nais_chains', type=int, default=10, help='')
parser.add_argument('-nais_steps', type=int, default=10000, help='')

parser.add_argument('-log_interval', type=int, default=200, help='report interval')
parser.add_argument('-save', type=str, default='', help='path to save the final model')
parser.add_argument('-train_from', type=str, default='', help='')
parser.add_argument('-nruns', type=int, default=100, help='')
parser.add_argument('-grid', action='store_true', help='')
parser.add_argument('-check_corr', action='store_true', help='')

def main(args, trdat, valdat, nvis, ne):
    print("main args", args)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    def get_infnet():
        if "justvis" in args.infarch:
            return SeqJustVisInfNet(nvis, args)
        elif "dbl" in args.infarch:
            return DblSeqInfNet(nvis, args)
        elif "2d" in args.infarch:
            return TwodJustVisInfNet(nvis, args)
        else:
            return SeqInfNet(nvis, args)

    if len(args.train_from) > 0:
        saved_stuff = torch.load(args.train_from)
        saved_args = saved_stuff["opt"]
        model = RBM(nvis, saved_args)
        model.load_state_dict(saved_stuff["mod_sd"])
        model = model.to(device)
        print("running ais...")
        valnpll, vnexagain = moar_validate(valdat, model, device, args, do_ais=True)
        print("Epoch {:3d} | val npll {:.3f}".format(
            0, valnpll/vnexagain))
        exit()
        # infnet = get_infnet()
        # infnet.load_state_dict(saved_stuff["inf_sd"])
        # infnet = infnet.to(device)
    else:
        model = RBM(nvis, args).to(device)
        infnet = get_infnet()
        infnet = infnet.to(device)

    ne = ne.to(device)

    if args.training == "lbp" or args.check_corr:
        edges = get_rbm_edges(nvis, args.nhid)#.to(device)

    bestmodel = RBM(nvis, args)
    bestinfnet = get_infnet()

    if args.penfunc == "l2":
        penfunc = lambda x, y: ((x-y)*(x-y)).sum(-1)
    elif args.penfunc == "l1":
        penfunc = lambda x, y: (x-y).abs().sum(-1)
    elif args.penfunc == "js":
        penfunc = lambda x, y: 0.5*(batch_kl(x, y) + batch_kl(y, x))
    elif args.penfunc == "kl1":
        penfunc = lambda x, y: batch_kl(x, y)
    elif args.penfunc == "kl2":
        penfunc = lambda x, y: batch_kl(y, x)
    else:
        penfunc = None

    best_loss, prev_loss = float("inf"), float("inf")
    lrdecay, pendecay = False, False
    if args.optalg == "sgd":
        popt1 = torch.optim.SGD(model.parameters(), lr=args.lr)
        popt2 = torch.optim.SGD(infnet.parameters(), lr=args.ilr)
    else:
        popt1 = torch.optim.Adam(model.parameters(), lr=args.lr)
        popt2 = torch.optim.Adam(infnet.parameters(), lr=args.ilr)


    if args.check_corr:
        from utils import corr

        nedges = nvis * model.nhid # V*H edges
        npenterms = 2 * nvis * model.nhid
        V, H = nvis, model.nhid
        with torch.no_grad():
            ed_lpots = model.get_edge_scores().unsqueeze(0) # 1 x nvis x nhid x 4
            un_lpots = model.get_unary_scores().unsqueeze(0) # 1 x (nvis + nhid) x 2
            exed_lpots = ed_lpots.view(nedges, 1, 2, 2) # V*H x 1 x 2 x 2
            exun_lpots = un_lpots.transpose(0, 1) # V+H x 1 x 2
            nodebs, facbs, _, _, _ = dolbp(exed_lpots, edges, x=None, emlps=exun_lpots,
                                           niter=args.lbp_iter, renorm=True, tol=args.lbp_tol,
                                           randomize=args.randomize_lbp)
            # reshape log unary marginals: V+H x 1 x 2 -> 1 x V+H x 2
            tau_u = torch.stack([nodebs[t] for t in range(nvis + model.nhid)]).transpose(0, 1)
            # reshape log fac marginals: nedges x 1 x 2 x 2 -> 1 x nedges x 2 x 2
            tau_e = torch.stack([facbs[e] for e in range(nedges)]).transpose(0, 1)
            # exponentiate
            tau_u, tau_e = (tau_u.exp() + EPS), (tau_e.exp() + EPS)

        for i in range(args.inf_iter):
            with torch.no_grad(): # these functions are used in calc'ing the loss below too
                pred_rho = infnet.q() # V*H x K^2 logits
                # should be 1 x nvis+nhid x 2 and 1 x V*H x K^2
                predtau_u, predtau_e, _ = get_taus_and_pens(pred_rho, V, H, penfunc=penfunc)
                predtau_u, predtau_e = predtau_u.exp() + EPS, predtau_e.exp() + EPS

            # i guess we'll just pick one entry from each
            un_margs = tau_u[0][:, 1] # V+H
            bin_margs = tau_e[0][:, 1, 1] # nedges
            pred_un_margs = predtau_u[0][:, 1] # T
            pred_bin_margs = predtau_e[0].view(-1, 2, 2)[:, 1, 1] # nedges
            # print("dufuq", tau_u.size(), predtau_u.size(), un_margs.size(), pred_un_margs.size())
            # print("srsly", tau_e.size(), predtau_e.size(), bin_margs.size(), pred_bin_margs.size())
            print(i, "unary corr: %.4f, binary corr: %.4f" %
                  (corr(un_margs, pred_un_margs),
                   corr(bin_margs, pred_bin_margs)))

            popt2.zero_grad()
            pred_rho = infnet.q() # V*H x K^2 logits
            in_loss, ipen_loss = rbm_inner_loss(pred_rho, un_lpots, ed_lpots, ne, penfunc)
            in_loss = in_loss + args.pen_mult/npenterms * ipen_loss
            print("in_loss", in_loss.item())
            in_loss.backward()
            clip_opt_params(popt2, args.clip)
            popt2.step()
        exit()

    bad_epochs = -1
    for ep in range(args.epochs):
        if args.training == "pcd":
            oloss, nex = train_pcd(trdat, model, popt1, device, args)
            print("Epoch {:3d} | train loss {:.3f}".format(
                ep, oloss/nex))
        elif args.training == "lbp":
            oloss, nex = lbp_train(trdat, model, popt1, edges, ne, device, args)
            print("Epoch {:3d} | train out_loss {:.3f}".format(ep, oloss/nex))
        else:
            oloss, iloss, ploss, nex = train(trdat, model, infnet, popt1, popt2,
                                             ne, penfunc, device, args)
            print("Epoch {:3d} | train out_loss {:.3f} | train in_loss {:.3f} | pen {:.3f}".format(
                ep, oloss/nex, iloss/nex, ploss/(nex*args.pen_mult)))

        with torch.no_grad():
            if args.training == "bethe":
                voloss, vploss, vnex = validate(valdat, model, infnet, ne, penfunc, device)
                print("Epoch {:3d} | val out_loss {:.3f} | val pen {:.3f}".format(
                    ep, voloss/vnex, vploss/(vnex*args.pen_mult)))
            elif args.training == "lbp":
                voloss, vnex = lbp_validate(valdat, model, edges, ne, device)
                print("Epoch {:3d} | val out_loss {:.3f}".format(
                    ep, voloss/vnex))

            trnpll, nexagain = moar_validate(trdat, model, device, args)
            print("Epoch {:3d} | train npll {:.3f}".format(
                ep, trnpll/nexagain))

            valnpll, vnexagain = moar_validate(valdat, model, device, args, do_ais=args.do_ais)
            print("Epoch {:3d} | val npll {:.3f}".format(
                ep, valnpll/vnexagain))
            voloss = valnpll

        if voloss < best_loss:
            best_loss = voloss
            bad_epochs = -1
            print("updating best model")
            bestmodel.load_state_dict(model.state_dict())
            bestinfnet.load_state_dict(infnet.state_dict())
            if len(args.save) > 0 and not args.grid:
                print("saving model to", args.save)
                torch.save({"opt": args, "mod_sd": bestmodel.state_dict(),
                            "inf_sd": bestinfnet.state_dict(), "bestloss": bestloss}, args.save)

        if (voloss >= prev_loss or lrdecay) and args.optalg == "sgd":
            for group in popt1.param_groups:
                group['lr'] *= args.lrdecay
            for group in popt2.param_groups:
                group['lr'] *= args.lrdecay
            #decay = True
        if (voloss >= prev_loss or pendecay):
            args.pen_mult *= args.pendecay
            print("pen_mult now", args.pen_mult)
            pendecay = True

        prev_loss = voloss
        if args.lr < 1e-5:
            break
        print("")
        bad_epochs += 1
        if bad_epochs >= 10:
            break
        # if args.reset_adam:
        #     print("resetting adam...")
        #     popt2 = torch.optim.Adam(infnet.parameters(), lr=args.ilr)
    return bestmodel, bestinfnet, best_loss


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.use_kuleshov_data:
        # Xtr, _, Xval, _, _, _ = data.load_digits()
        # Xtr = torch.from_numpy(Xtr).squeeze(1).round() # N x dim1 x dim2
        # Xval = torch.from_numpy(Xval).squeeze(1).round() # N x dim1 x dim2
        Xtr, Xval = torch.load("kuleshov_digits.pt") # each is N x dim1 x dim2; in [0, 1]
        Xtr, Xval = Xtr.round(), Xval.round()
        nvis = Xtr.size(1) * Xtr.size(2)
    else:
        with open(os.path.join(args.bin_mnist_dir, "binarized_mnist_train.amat")) as f:
            Xtr = torch.stack([torch.Tensor([int(v) for v in line.strip().split()]) for line in f])

        with open(os.path.join(args.bin_mnist_dir, "binarized_mnist_valid.amat")) as f:
            Xval = torch.stack([torch.Tensor([int(v) for v in line.strip().split()]) for line in f])

        nvis = Xtr.size(1)

    print("Xtr size", Xtr.size())

    trdat = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr), batch_size=args.bsz, shuffle=True)
    valdat = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xval), batch_size=args.bsz, shuffle=False)


    ne = get_rbm_ne(nvis, args.nhid) # 1 x V+H x 2 (last dimension is just repeated)
    ne = ne + 1 # account for unary factors

    #assert False

    def get_pms(opts):
        if opts.penfunc == "l2":
            #pms = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
            pms = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        else:
            #pms = [0.1, 0.5, 1, 5, 10]
            pms = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        return pms

    def get_q_hidsz(opts):
        if "rnn" in opts.infarch:
            return [64, 100, 200, 300, 600]
        if "justvis" in opts.infarch or "dbl" in opts.infarch:
            return [opts.qemb_sz]
        # just for non-rnn single sequence....
        return [2*opts.qemb_sz]

    # hypers = OrderedDict({'optalg': ['adam'],
    #                       'pendecay': [1, 0.9, 0.7, 0.5],
    #                       'penfunc': ["l2", "js", "kl2", "kl1"],
    #                       #"infarch": ["yoon", "transformer", "yoonjustvis", "transformerjustvis"],
    #                       #'infarch': ["rnn", "rnnjustvis"],
    #                       #'infarch': ["cnn", "cnnjustvis"],
    #                       #'infarch': ["rnn", "rnndbl"],
    #                       'infarch': ["transformer", "transformerdbl"],
    #                       'init_strat': ["xavier", "gaussian", "uniform"],
    #                       'lr': [0.003, 0.001, 0.0003, 0.0001, 0.00003],
    #                       'ilr': [0.003, 0.001, 0.0003, 0.0001, 0.00003],
    #                       'qemb_sz': [64, 100, 150, 200, 300],
    #                       'q_hid_size': get_q_hidsz,
    #                       'q_layers': list(range(1, 6)),
    #                       'q_heads': list(range(1, 6)),
    #                       'clip': [1, 5, 10],
    #                       'pen_mult': get_pms,
    #                       'seed': list(range(100000)),
    #                       'init': [0.1, 0.05, 0.01],
    #                       'qinit': [0.1, 0.05, 0.01],
    #                      })


    hypers = OrderedDict({'optalg': ['adam'],
                          'pendecay': [1, 0.9, 0.7, 0.5],
                          'penfunc': ["l2", "js", "kl2", "kl1"],
                          'infarch': ["transformer", "transformerdbl"],
                          'init_strat': ["xavier", "gaussian"],
                          'lr': [0.005, 0.003, 0.001, 0.0005, 0.0001],
                          'ilr': [0.005, 0.003, 0.001, 0.0005, 0.0001],
                          'qemb_sz': [64, 100, 150],
                          'q_hid_size': get_q_hidsz,
                          'q_layers': [1, 2, 3], #list(range(1, 6)),
                          'q_heads': [1, 2, 3], #list(range(1, 6)),
                          'clip': [1, 5],
                          'pen_mult': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
                          'seed': list(range(100000)),
                          'init': [0.1, 0.05],
                          'qinit': [0.1],
                          'reset_adam': [True, False],
                          'inf_iter': [1, 10, 20, 40, 50],
                         })

    torch.manual_seed(args.seed)

    if not args.grid:
        args.nruns = 1

    bestloss = float("inf")
    for _ in range(args.nruns):
        if args.grid:
            for hyp, choices in hypers.items():
                if isinstance(choices, list):
                    hypvals = choices
                else: # it's a function
                    hypvals = choices(args)
                choice = hypvals[torch.randint(len(hypvals), (1,)).item()]
                args.__dict__[hyp] = choice

        #try:
        bestmodel, bestinfnet, runloss = main(args, trdat, valdat, nvis, ne)
        # except RuntimeError: # presumably oom
        #     runloss = float("inf")
        #     torch.cuda.empty_cache()
        #     print("skipping b/c of oom")
        if runloss < bestloss:
            bestloss = runloss
            if len(args.save) > 0:
                print("saving model to", args.save)
                torch.save({"opt": args, "mod_sd": bestmodel.state_dict(),
                            "inf_sd": bestinfnet.state_dict(), "bestloss": bestloss}, args.save)
        print()
        print()
