import os
import math
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn

from uhmm_infnets import RNodeInfNet, TNodeInfNet
from utils import InfcHelper, clip_opt_params, get_hmm_stuff, bethe_fex, bethe_fez, batch_kl
import data
from infc_utils import batch_ufwdalg, exact_marginals
from lbp_util import dolbp

class HybEdgeModel(nn.Module):
    def __init__(self, ntypes, max_verts, opt):
        super(HybEdgeModel, self).__init__()
        self.K, self.M = opt.K, opt.markov_order
        self.drop = nn.Dropout(opt.dropout)

        self.lembs = nn.Parameter(torch.Tensor(opt.K, opt.lemb_size))

        # stuff for edge embeddings
        self.just_diff, self.use_length = opt.just_diff, opt.use_length
        self.with_idx_ind = opt.with_idx_ind
        self.max_verts = max_verts + 1
        vrows = self.max_verts + self.with_idx_ind*(max(1, self.M-1)+1)
        self.vlut = nn.Embedding(vrows, opt.vemb_size)
        self.tlembs = nn.Parameter(torch.Tensor(opt.K, opt.lemb_size))

        tdec_inp_size = (opt.vemb_size + (opt.use_length)*opt.vemb_size
                         + opt.lemb_size)

        self.resid = not opt.not_residual

        if self.resid:
            self.decoder = nn.Linear(opt.lemb_size, ntypes)
            self.em_mlp = nn.Sequential(nn.Linear(opt.lemb_size, opt.lemb_size),
                                        nn.ReLU(), self.drop)
            self.em_norm = nn.LayerNorm(opt.lemb_size)

            self.trans_mlp = nn.Sequential(nn.Linear(tdec_inp_size, tdec_inp_size),
                                           nn.ReLU())
            self.trans_norm = nn.LayerNorm(tdec_inp_size)
            self.trans_decoder = nn.Linear(tdec_inp_size, opt.K)
        else:
            self.decoder = nn.Sequential(nn.Linear(opt.lemb_size, opt.wemb_size), nn.ReLU(),
                                         self.drop, nn.Linear(opt.wemb_size, ntypes))
            self.trans_mlp = nn.Sequential(nn.Linear(tdec_inp_size, opt.t_hid_size),
                                           nn.ReLU())
            self.trans_decoder = nn.Linear(opt.t_hid_size, opt.K)

        self.pinit = opt.init
        self.init_weights()

    def init_weights(self):
        initrange = self.pinit
        if self.resid:
            lins = [self.decoder, self.em_mlp[0]]
        else:
            lins = [self.decoder[0], self.decoder[-1]]
        lins.extend([self.trans_decoder, self.trans_mlp[0], self.vlut])
        for lin in lins:
            lin.weight.data.uniform_(-initrange, initrange)
            if hasattr(lin, "bias"):
                lin.bias.data.zero_()

        params = [self.lembs, self.tlembs]
        for par in params:
            par.data.uniform_(-initrange, initrange)

    def get_emdist(self):
        if self.resid:
            emdist = torch.log_softmax(
                self.decoder(self.em_norm(self.lembs + self.em_mlp(self.lembs))), dim=1) # K x V
        else:
            emdist = torch.log_softmax(self.decoder(self.lembs), dim=1) # K x V
        return emdist

    def get_obs_lps(self, x):
        """
        x - T x bsz
        returns bsz x T x K unary potentials
        """
        T, bsz = x.size()
        emdist = self.get_emdist() # K x V
        # select class-potentials for each observed word
        emlps = emdist.t()[x] # T x bsz x K
        return emlps.transpose(0, 1).contiguous()

    def get_edge_scores(self, e, T):
        """
        e - nedges x 2, where each entry is a 0-indexed symbol. e.g., [0, 1] might
            represent edge 0 <-> 1. we don't take a batch b/c we assume same structure
            for entire batch (and would be annoying otherwise).
        returns nedges x K*K log potentials
        """
        nedges = e.size(0)
        if self.just_diff: # equivalent to standard homogenous hmm stuff
            mye = (e[:, :1] - e[:, 1:]).abs() # nedges x 1
            eembs = self.vlut(mye).squeeze(1) # nedges x 1 x emb_size -> nedges x embsize
        elif self.with_idx_ind:
            # this is maybe cheatingish?
            mye = (e[:, :1] - e[:, 1:]).abs() # nedges x 1
            # get indicator for edges ending thru max(1, M-1) for fair comparison w/ directed
            ende = torch.clamp(e[:, 1:] + (self.max_verts-1), 0, self.vlut.num_embeddings-1)
            #eembs = torch.cat([self.vlut(mye).squeeze(1), self.vlut(ende).squeeze(1)], 1)
            eembs = self.vlut(mye).squeeze(1) + self.vlut(ende).squeeze(1)
        else:
            eembs = self.vlut(e) # nedges x 2 x embsize
            # eembs = eembs.sum(1) # maybe better to sum, max, or mlp?
            eembs = eembs.view(nedges, -1) # nedges x 2*embsize

        if self.use_length:
            # concatenate on length for nedges x 2*emb_size (or x 3*emb_size)
            eembs = torch.cat([eembs, self.lut.weight[T].view(1, -1).expand(nedges, -1)], 1)

        # concatenate each edge rep with each label rep to get something
        # nedges*K x (eembs.size(1) + tlembs.size(1))
        expedges = eembs.view(nedges, 1, -1).repeat(1, self.K, 1)
        edbylabe = torch.cat([expedges.view(nedges*self.K, -1),
                              self.tlembs.repeat(nedges, 1)], 1)

        # now get nedges*K x K transition scores
        if self.resid:
            trans_reps = self.trans_norm(edbylabe + self.trans_mlp(edbylabe))
        else:
            trans_reps = self.trans_mlp(edbylabe)

        if self.training:
            # since we're tying edges we don't want different dropout masks for the same
            # kind of edge (e.g., for just_diff), so use a single mask
            onerate = 1 - self.drop.p
            mask = torch.bernoulli(torch.full_like(trans_reps[:1], onerate))/onerate # 1 x repsize
            tscores = self.trans_decoder(trans_reps * mask)
        else:
            tscores = self.trans_decoder(trans_reps)

        return tscores.view(nedges, self.K*self.K)


def get_pred_lmargs(pw_logits, K, nodeidxs, neginf):
    """
    pw_logits - bsz x nedges*K*K
    nodeidxs - T x maxne, where we consider the nedges outgoing then nedges incoming edges
    neginf - 1 x 1 x 1
    returns:
      bsz x nedges x K^2 predicted factor log marginals
      bsz x 2*nedges+1 x K predicted row then column log margs
      bsz x T x K averaged log margs
    """
    bsz, nedges = pw_logits.size(0), pw_logits.size(1)//(K*K)
    T = nodeidxs.size(0)
    all_pw_logZs = pw_logits.view(bsz, nedges, -1).logsumexp(2).view(bsz, -1, 1) # bsz x nedges x 1
    pred_rowlmargs = pw_logits.view(bsz, nedges, K, K).logsumexp(3) # bsz x nedges x K
    pred_collmargs = pw_logits.view(bsz, nedges, K, K).logsumexp(2) # bsz x nedges x K
    # log normalize so we get row and column log marginals
    pred_rowlmargs = pred_rowlmargs - all_pw_logZs
    pred_collmargs = pred_collmargs - all_pw_logZs
    # average to get marginals by node
    nne = (nodeidxs != 2*nedges).sum(1).float().view(1, T, 1) # nedges is a dummy value
    flatnodeidxs = nodeidxs.view(1, -1, 1) # 1 x T*maxne x 1 idxs
    cat_lmargs = torch.cat( # bsz x 2*nedges + 1 x K
        [pred_rowlmargs, pred_collmargs, neginf.expand(bsz, 1, K)], 1)
    nidxs = flatnodeidxs.size(1) # T*maxne
    avlmargs = (cat_lmargs.gather( # average over all incoming and outgoing edges
        1, flatnodeidxs.expand(bsz, nidxs, K)).view(bsz, T, -1, K).logsumexp(2)
                - nne.expand(bsz, T, K).log())
    # normalize factor marginals before returning
    predfac_lmargs = pw_logits.view(bsz, nedges, -1) - all_pw_logZs

    return predfac_lmargs, cat_lmargs, avlmargs


def get_lmarg_penalties(predcat_lmargs, avlmargs, nodeidxs, penfunc):
    """
    predcat_lmargs - bsz x 2*nedges+1 x K
    avlmargs - bsz x T x K
    nodeidxs - T x maxne, where we consider the nedges outgoing then nedges incoming edges
    """
    bsz, _, K = predcat_lmargs.size()
    maxne = nodeidxs.size(1)
    dummyidx = predcat_lmargs.size(1)-1 # 2*nedges
    mask = (nodeidxs != dummyidx).float() # T x maxne mask
    # make 1 x T*maxne x 1 idxs
    flatnodeidxs = nodeidxs.view(1, -1, 1)
    nidxs = flatnodeidxs.size(1)
    T = avlmargs.size(1)
    # do penalties; get bsz x T x maxne x K predicted marginals
    exlmargs = predcat_lmargs.gather(1, flatnodeidxs.expand(bsz, nidxs, K)).view(bsz, T, -1, K)
    pens = penfunc(exlmargs, avlmargs.view(bsz, T, 1, K).expand(bsz, T, maxne, K))
    # zero out rows that are just padding
    pens = (pens * mask.view(1, T, maxne).expand(bsz, T, maxne)).sum()
    return pens


def get_taus_and_pens(rho, nodeidxs, K, neginf, penfunc=None):
    """
    rho - bsz x nedges*K*K
    returns bsz x T x K avg node lmargs, bsz x nedges x K^2 predicted fac lmargs, penloss
    """
    predfac_lmargs, cat_lmargs, avlmargs = get_pred_lmargs(rho, K, nodeidxs, neginf)
    if penfunc is not None:
        tgtunaries = avlmargs
        penloss = get_lmarg_penalties(cat_lmargs, tgtunaries, nodeidxs, penfunc)
    else:
        penloss = 0
    return avlmargs, predfac_lmargs, penloss

EPS = 1e-38

def outer_loss(rho_x, rho, un_lpots, ed_lpots, nodeidxs, K, ne, neginf, penfunc):
    """
    ne - 1 x nnodes*K
    un_lpots - bsz x nnodes*K
    ed_lpots - 1 x nedges*K*Ks
    """
    # the full loss: F_x(tau_x, theta) - F(tau, theta)
    # note that ed_lpots is the same as lpots but w/ no observation factors
    bsz = rho_x.size(0)
    taux_u, taux_e, penx = get_taus_and_pens(rho_x, nodeidxs, K, neginf, penfunc=penfunc)
    taux_u, taux_e = taux_u.exp() + EPS, taux_e.exp() + EPS
    tau_u, tau_e, pen = get_taus_and_pens(rho, nodeidxs, K, neginf, penfunc=penfunc)
    tau_u, tau_e = tau_u.exp() + EPS, tau_e.exp() + EPS
    fx, _, _, _ = bethe_fex(taux_u.view(bsz, -1), taux_e.view(bsz, -1), un_lpots,
                            ed_lpots.expand(bsz, -1), ne.expand(bsz, -1))
    fz, _, _, _ = bethe_fez(tau_u.view(1, -1), tau_e.view(1, -1), ed_lpots, ne)
    return fx - fz*bsz, penx + pen*bsz

def inner_lossz(rho, ed_lpots, nodeidxs, K, ne, neginf, penfunc):
    """
    rho - 1 x nedges*K*K
    ed_lpots - 1 x nedges*K*K
    ne - 1 x nnodes*K
    """
    # inner loss is minus the outer_loss wrt tau, which is equivalent to just F(tau, theta)
    tau_u, tau_e, penloss = get_taus_and_pens(rho, nodeidxs, K, neginf, penfunc=penfunc)
    tau_u, tau_e = tau_u.exp() + EPS, tau_e.exp() + EPS
    fz, _, _, _ = bethe_fez(tau_u.view(1, -1), tau_e.view(1, -1), ed_lpots, ne)
    return fz, penloss

def inner_lossx(rho, un_lpots, ed_lpots, nodeidxs, K, ne, neginf, penfunc):
    """
    rho - bsz x nedges*K*K
    un_lpots - bsz x nnodes*K
    ed_lpots - bsz x nedges*K*K
    ne - 1 x nnodes*K
    """
    # inner loss is minus the outer_loss wrt tau, which is equivalent to just F(tau, theta)
    bsz = un_lpots.size(0)
    tau_u, tau_e, penloss = get_taus_and_pens(rho, nodeidxs, K, neginf, penfunc=penfunc)
    tau_u, tau_e = tau_u.exp() + EPS, tau_e.exp() + EPS
    fx, _, _, _ = bethe_fex(tau_u.view(bsz, -1), tau_e.view(bsz, -1), un_lpots, ed_lpots,
                            ne.expand(bsz, -1))
    return fx, penloss


# alternates minimizing over rho_x, theta, and maxing over rho
def train_unsup_am(corpus, model, infnet, moptim, ioptim, cache, penfunc, neginf, device, args):
    """
    opt is just gonna be for everyone
    """
    model.train()
    infnet.train()
    K, M = args.K, args.markov_order
    total_out_loss, total_in_loss, nexamples = 0.0, 0.0, 0
    total_pen_loss = 0.0
    perm = torch.randperm(len(corpus))
    for i, idx in enumerate(perm):
        batch = corpus[idx.item()].to(device)
        T, bsz = batch.size()
        if T <= 1: # annoying
            continue
        if T not in cache:
            edges, nodeidxs, ne = get_hmm_stuff(T, M, K)
            cache[T] = (edges, nodeidxs, ne)
        edges, nodeidxs, ne = cache[T]
        edges = edges.to(device) # symbolic edge representation
        ne, nodeidxs = ne.view(1, -1).to(device), nodeidxs.to(device) # 1 x T*K, # T x maxne
        npenterms = (nodeidxs != 2*edges.size(0)).sum().float()

        # maximize wrt rho
        with torch.no_grad():
            ed_lpots = model.get_edge_scores(edges, T) # nedges x K*K log potentials

        # if args.reset_adam:
        #     ioptim = torch.optim.Adam(infnet.parameters(), lr=args.ilr)

        for _ in range(args.z_iter):
            ioptim.zero_grad()
            pred_rho = infnet.q(edges, T) # nedges x K^2 logits
            in_loss, ipen_loss = inner_lossz(pred_rho.view(1, -1), ed_lpots.view(1, -1), nodeidxs,
                                             K, ne, neginf, penfunc)
            total_in_loss += in_loss.item()*bsz
            total_pen_loss += args.pen_mult/npenterms * ipen_loss.item()*bsz
            in_loss = in_loss + args.pen_mult/npenterms * ipen_loss
            in_loss.backward()
            clip_opt_params(ioptim, args.clip)
            ioptim.step()

        pred_rho = pred_rho.detach()

        if args.loss == "alt3":
            # min wrt rho_x
            with torch.no_grad():
                un_lpots = model.get_obs_lps(batch) # bsz x T x K log unary potentials

            for _ in range(args.zx_iter):
                ioptim.zero_grad()
                pred_rho_x = infnet.qx(batch, edges, T)
                out_loss1, open_loss1 = inner_lossx(
                    pred_rho_x, un_lpots.view(bsz, -1), ed_lpots.view(1, -1).expand(bsz, -1),
                    nodeidxs, K, ne.expand(bsz, -1), neginf, penfunc)
                out_loss1 = out_loss1 + args.pen_mult/npenterms * open_loss1
                total_pen_loss += args.pen_mult/npenterms * open_loss1.item()
                out_loss1.div(bsz).backward()
                clip_opt_params(ioptim, args.clip)
                ioptim.step()
            pred_rho_x = pred_rho_x.detach()

        # min wrt params
        moptim.zero_grad()
        # even tho these don't change we needa do it again
        un_lpots = model.get_obs_lps(batch) # bsz x T x K log unary potentials
        ed_lpots = model.get_edge_scores(edges, T) # nedges x K*K log potentials

        if args.loss != "alt3": # jointly minimizing over rho_x
            pred_rho_x = infnet.qx(batch, edges, T)
            openfunc = penfunc
        else:
            openfunc = None

        out_loss, open_loss = outer_loss(pred_rho_x, pred_rho, un_lpots.view(bsz, -1),
                                         ed_lpots.view(1, -1), nodeidxs, K, ne, neginf,
                                         penfunc=openfunc)
        total_out_loss += out_loss.item()
        if args.loss != "alt3":
            total_pen_loss += args.pen_mult/npenterms * open_loss
        out_loss = out_loss + args.pen_mult/npenterms * open_loss
        out_loss.div(bsz).backward()
        clip_opt_params(moptim, args.clip)
        moptim.step()
        nexamples += bsz

        if (i+1) % args.log_interval == 0:
            print("{:5d}/{:5d} | out_loss {:8.5f} | in_loss {:8.5f} | pen_loss {:8.6f}".format(
                i+1, perm.size(0), total_out_loss/nexamples, total_in_loss/nexamples,
                total_pen_loss/(nexamples*args.pen_mult)))

    return total_out_loss, total_in_loss, total_pen_loss, nexamples



def validate_unsup_am(corpus, model, infnet, cache, penfunc, neginf, device, args):
    model.eval()
    infnet.eval()
    K, M = args.K, args.markov_order
    total_out_loss, total_pen_loss, nexamples = 0.0, 0.0, 0

    for i in range(len(corpus)):
        batch = corpus[i].to(device)
        T, bsz = batch.size()
        if T <= 1: # annoying
            continue
        if T not in cache:
            edges, nodeidxs, ne = get_hmm_stuff(T, M, K)
            cache[T] = (edges, nodeidxs, ne)
        edges, nodeidxs, ne = cache[T]
        edges = edges.to(device) # symbolic edge representation
        ne, nodeidxs = ne.view(1, -1).to(device), nodeidxs.to(device) # 1 x T*K, # T x maxne
        npenterms = (nodeidxs != 2*edges.size(0)).sum().float()

        # maximize wrt rho
        ed_lpots = model.get_edge_scores(edges, T) # nedges x K*K log potentials
        pred_rho = infnet.q(edges, T)

        un_lpots = model.get_obs_lps(batch) # bsz x T x K log unary potentials
        pred_rho_x = infnet.qx(batch, edges, T)

        out_loss, open_loss = outer_loss(pred_rho_x, pred_rho, un_lpots.view(bsz, -1),
                                         ed_lpots.view(1, -1), nodeidxs, K, ne, neginf,
                                         penfunc=penfunc)
        total_out_loss += out_loss.item()
        total_pen_loss += 1/npenterms * open_loss.item()
        nexamples += bsz

    return total_out_loss, total_pen_loss, nexamples


def lbp_train(corpus, model, popt, helper, device, args):
    model.train()
    K, M = args.K, helper.markov_order
    total_loss, nexamples = 0.0, 0
    niter, nxiter = 0, 0
    perm = torch.randperm(len(corpus))
    for i, idx in enumerate(perm):
        popt.zero_grad()
        batch = corpus[idx.item()].to(device)
        T, bsz = batch.size()
        # if T <= 1 or (M == 3 and T <= 3): # annoying
        #     continue
        if T <= 1:
            continue
        if T not in cache:
            edges, nodeidxs, ne = get_hmm_stuff(T, M, K)
            cache[T] = (edges, nodeidxs, ne)
        edges, nodeidxs, ne = cache[T]
        nedges = edges.size(0)
        edges, ne = edges.to(device), ne.view(1, -1).to(device)

        un_lpots = model.get_obs_lps(batch) # bsz x T x K log unary potentials
        ed_lpots = model.get_edge_scores(edges, T) # nedges x K*K log potentials

        with torch.no_grad():
            exed_lpots = ed_lpots.view(nedges, 1, K, K)
            # get approximate unclamped marginals
            nodebs, facbs, ii, _, _ = dolbp(exed_lpots, edges, niter=args.lbp_iter, renorm=True,
                                            randomize=args.randomize_lbp, tol=args.lbp_tol)
            xnodebs, xfacbs, iix, _, _ = dolbp(exed_lpots.expand(nedges, bsz, K, K), edges,
                                               x=batch, emlps=un_lpots.transpose(0, 1),
                                               niter=args.lbp_iter, renorm=True,
                                               randomize=args.randomize_lbp, tol=args.lbp_tol)
            niter += ii
            nxiter += iix
            # reshape log unary marginals: T x bsz x K -> bsz x T x K
            tau_u = torch.stack([nodebs[t] for t in range(T)]).transpose(0, 1)
            taux_u = torch.stack([xnodebs[t] for t in range(T)]).transpose(0, 1)
            # reshape log fac marginals: nedges x bsz x K x K -> bsz x nedges x K x K
            tau_e = torch.stack([facbs[e] for e in range(nedges)]).transpose(0, 1)
            taux_e = torch.stack([xfacbs[e] for e in range(nedges)]).transpose(0, 1)

            # exponentiate
            tau_u, tau_e = (tau_u.exp() + EPS).view(1, -1), (tau_e.exp() + EPS).view(1, -1)
            taux_u, taux_e = (taux_u.exp() + EPS).view(bsz, -1), (taux_e.exp() + EPS).view(bsz, -1)

        fx, _, _, _ = bethe_fex(taux_u, taux_e, un_lpots.view(bsz, -1),
                                ed_lpots.view(1, -1).expand(bsz, -1), ne.expand(bsz, -1))
        fz, _, _, _ = bethe_fez(tau_u, tau_e, ed_lpots.view(1, -1), ne)
        loss = fx - fz*bsz
        total_loss += loss.item()
        loss.div(bsz).backward()
        clip_opt_params(popt, args.clip)
        popt.step()
        nexamples += bsz

        if (i+1) % args.log_interval == 0:
            print("{:5d}/{:5d} | its {:3.2f}/{:3.2f} | out_loss {:8.3f}".format(
                i+1, perm.size(0), niter/(i+1), nxiter/(i+1), total_loss/nexamples))

    return total_loss, nexamples

def lbp_validate(corpus, model, helper, device):
    model.eval()
    K, M = args.K, helper.markov_order
    total_loss, nexamples = 0.0, 0
    for i in range(len(corpus)):
        batch = corpus[i].to(device)
        T, bsz = batch.size()
        if T <= 1:
            continue
        if T not in cache:
            edges, nodeidxs, ne = get_hmm_stuff(T, M, K)
            cache[T] = (edges, nodeidxs, ne)
        edges, nodeidxs, ne = cache[T]
        nedges = edges.size(0)
        edges, ne = edges.to(device), ne.view(1, -1).to(device)

        un_lpots = model.get_obs_lps(batch) # bsz x T x K log unary potentials
        ed_lpots = model.get_edge_scores(edges, T) # nedges x K*K log potentials

        exed_lpots = ed_lpots.view(nedges, 1, K, K)
        # get approximate unclamped marginals
        nodebs, facbs, _, _, _ = dolbp(exed_lpots, edges, niter=args.lbp_iter,
                                       renorm=True, randomize=args.randomize_lbp, tol=args.lbp_tol)
        xnodebs, xfacbs, _, _, _ = dolbp(exed_lpots.expand(nedges, bsz, K, K), edges, x=batch,
                                         emlps=un_lpots.transpose(0, 1), niter=args.lbp_iter,
                                         renorm=True, randomize=args.randomize_lbp,
                                         tol=args.lbp_tol)
        # reshape log unary marginals: T x bsz x K -> bsz x T x K
        tau_u = torch.stack([nodebs[t] for t in range(T)]).transpose(0, 1)
        taux_u = torch.stack([xnodebs[t] for t in range(T)]).transpose(0, 1)
        # reshape log fac marginals: nedges x bsz x K x K -> bsz x nedges x K x K
        tau_e = torch.stack([facbs[e] for e in range(nedges)]).transpose(0, 1)
        taux_e = torch.stack([xfacbs[e] for e in range(nedges)]).transpose(0, 1)

        # exponentiate
        tau_u, tau_e = (tau_u.exp() + EPS).view(1, -1), (tau_e.exp() + EPS).view(1, -1)
        taux_u, taux_e = (taux_u.exp() + EPS).view(bsz, -1), (taux_e.exp() + EPS).view(bsz, -1)

        fx, _, _, _ = bethe_fex(taux_u, taux_e, un_lpots.view(bsz, -1),
                                ed_lpots.view(1, -1).expand(bsz, -1), ne.expand(bsz, -1))
        fz, _, _, _ = bethe_fez(tau_u, tau_e, ed_lpots.view(1, -1), ne)
        loss = fx - fz*bsz
        total_loss += loss.item()
        nexamples += bsz

    return total_loss, nexamples


def exact_train(corpus, model, optim, helper, device, args):
    model.train()
    ll, ntokens = 0.0, 0
    perm = torch.randperm(len(corpus))
    for i, idx in enumerate(perm):
        # if i > 1:
        #     break
        optim.zero_grad()
        batch = corpus[idx.item()].to(device)
        T, bsz = batch.size()
        # get normalizer; depends only on edges
        if T > 1:
            edges = helper.get_edges(T).to(device) # symbolic edge representation
            edge_scores = model.get_edge_scores(edges, T) # nedges x K*K
            ending_at = helper.get_ending_at(T) # used to make infc less annoying
            lnZ = batch_ufwdalg(edge_scores, T, helper.K, helper.markov_order, edges,
                                ending_at=ending_at) # 1
        else:
            edges, edge_scores, ending_at, lnZ = None, None, 0, 0
        # get unnormalized marginal
        obs_lpots = model.get_obs_lps(batch) # bsz x T x K
        lnZx = batch_ufwdalg(edge_scores, T, helper.K, helper.markov_order, edges,
                             ulpots=obs_lpots, ending_at=ending_at) # bsz
        logmarg = (lnZx - lnZ).sum()
        ll += logmarg.item()
        logmarg.div(-bsz).backward()
        ntokens += batch.nelement()
        clip_opt_params(optim, args.clip)
        optim.step()
        if (i+1) % args.log_interval == 0:
            print("{:5d}/{:5d} | lr {:02.4f} | ppl {:8.2f}".format(
                i+1, perm.size(0), args.lr, math.exp(-ll/ntokens)))
    return ll, ntokens

def exact_validate(corpus, model, helper, device):
    model.eval()
    ll, ntokens = 0.0, 0
    for i in range(len(corpus)):
        batch = corpus[i].to(device)
        T, bsz = batch.size()
        if T > 1:
            edges = helper.get_edges(T).to(device) # symbolic edge representation
            edge_scores = model.get_edge_scores(edges, T) # nedges x K*K
            edge_scores = edge_scores.double() # so we don't get dumb overflow shit
            ending_at = helper.get_ending_at(T) # used to make infc less annoying
            lnZ = batch_ufwdalg(edge_scores, T, helper.K, helper.markov_order, edges,
                                ending_at=ending_at) # 1
        else:
            edges, edge_scores, ending_at, lnZ = None, None, 0, 0
        # get unnormalized marginal
        obs_lpots = model.get_obs_lps(batch) # bsz x T x K
        obs_lpots = obs_lpots.double() # overflow, etc
        lnZx = batch_ufwdalg(edge_scores, T, helper.K, helper.markov_order, edges,
                             ulpots=obs_lpots, ending_at=ending_at) # bsz
        logmarg = (lnZx - lnZ).sum()
        ll += logmarg.item()
        ntokens += batch.nelement()
    return ll, ntokens

parser = argparse.ArgumentParser(description='')
parser.add_argument('-data', type=str, default="/scratch/data/ptb/",
                    help='location of the data corpus')
parser.add_argument('-thresh', type=int, default=0, help='')
parser.add_argument('-max_len', type=int, default=50, help='')

parser.add_argument('-not_residual', action='store_true', help='')
parser.add_argument('-wemb_size', type=int, default=100, help='size of word embeddings')
parser.add_argument('-lemb_size', type=int, default=100, help='size of latent label embeddings')
parser.add_argument('-vemb_size', type=int, default=100, help='size of index embeddings')
parser.add_argument('-not_inf_residual', action='store_true', help='')

parser.add_argument('-q_hid_size', type=int, default=100, help='')
parser.add_argument('-q_layers', type=int, default=2, help='')
parser.add_argument('-q_heads', type=int, default=3, help='')
parser.add_argument('-qemb_size', type=int, default=100, help='size of inf index embeddings')
parser.add_argument('-infarch', type=str, default='transnode',
                    choices=["rnnnode", "transnode"], help='')

parser.add_argument('-just_diff', action='store_true', help='potentials just look at abs(diff)')
parser.add_argument('-use_length', action='store_true', help='use length as potential feature')
parser.add_argument('-with_idx_ind', action='store_true', help='feature for edges ending at max(1, M-1)')
parser.add_argument('-t_hid_size', type=int, default=100, help='transition hid size')

parser.add_argument('-K', type=int, default=45, help='')
parser.add_argument('-markov_order', type=int, default=1, help='')

parser.add_argument('-optalg', type=str, default='sgd', choices=["sgd", "adam"], help='')
parser.add_argument('-loss', type=str, default='alt3',
                    choices=["exact", "lbp", "alt2", "alt3"], help='')
parser.add_argument('-pen_mult', type=float, default=1, help='')
parser.add_argument('-penfunc', type=str, default="l2",
                    choices=["l2", "l1", "js", "kl1", "kl2"], help='')
parser.add_argument('-pendecay', type=float, default=1, help='initial learning rate')
parser.add_argument('-lbp_iter', type=int, default=10, help='')
parser.add_argument('-lbp_tol', type=float, default=0.001, help='')
parser.add_argument('-randomize_lbp', action='store_true', help='')
parser.add_argument('-reset_adam', action='store_true', help='')
parser.add_argument('-z_iter', type=int, default=1, help='')
parser.add_argument('-zx_iter', type=int, default=1, help='')

parser.add_argument('-init', type=float, default=0.1, help='param init')
parser.add_argument('-qinit', type=float, default=0.1, help='infnet param init')
parser.add_argument('-lr', type=float, default=1, help='initial learning rate')
parser.add_argument('-ilr', type=float, default=1, help='initial learning rate')
parser.add_argument('-decay', type=float, default=0.5, help='initial learning rate')
parser.add_argument('-clip', type=float, default=5, help='gradient clipping')
parser.add_argument('-epochs', type=int, default=40, help='upper epoch limit')
parser.add_argument('-bsz', type=int, default=8, help='batch size')
parser.add_argument('-dropout', type=float, default=0.2, help='dropout')
parser.add_argument('-seed', type=int, default=1111, help='random seed')
parser.add_argument('-cuda', action='store_true', help='use CUDA')

parser.add_argument('-log_interval', type=int, default=200, help='report interval')
parser.add_argument('-save', type=str, default='', help='path to save the final model')
parser.add_argument('-train_from', type=str, default='', help='')
parser.add_argument('-nruns', type=int, default=100, help='random seed')
parser.add_argument('-grid', action='store_true', help='')
parser.add_argument('-check_corr', action='store_true', help='')

def main(args, helper, cache, max_seqlen, max_verts, ntypes, trbatches, valbatches):
    print("main args", args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.infarch == "rnnnode":
        infctor = RNodeInfNet
    else:
        infctor = TNodeInfNet

    model = HybEdgeModel(ntypes, max_verts, args).to(device)
    if "exact" not in args.loss:
        infnet = infctor(ntypes, max_seqlen, args).to(device)

    bestmodel = HybEdgeModel(ntypes, max_verts, args)
    if "exact" not in args.loss:
        bestinfnet = infctor(ntypes, max_seqlen, args)
    else:
        bestinfnet = None

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

    neginf = torch.Tensor(1, 1, 1).fill_(-1e18).to(device)

    best_loss, prev_loss = float("inf"), float("inf")
    lrdecay, pendecay = False, False
    if "exact" in args.loss:
        if args.optalg == "sgd":
            popt1 = torch.optim.SGD(model.parameters(), lr=args.lr)
        else:
            popt1 = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        if args.optalg == "sgd":
            popt1 = torch.optim.SGD(model.parameters(), lr=args.lr)
            popt2 = torch.optim.SGD(infnet.parameters(), lr=args.ilr)
        else:
            popt1 = torch.optim.Adam(model.parameters(), lr=args.lr)
            popt2 = torch.optim.Adam(infnet.parameters(), lr=args.ilr)

    if args.check_corr:
        from utils import corr
        # pick a graph to check
        T, K = 10, args.K
        edges, nodeidxs, ne = get_hmm_stuff(T, args.markov_order, K)
        edges, ne = edges.to(device), ne.view(1, -1).to(device)
        nodeidxs = nodeidxs.to(device)
        npenterms = (nodeidxs != 2*edges.size(0)).sum().float()
        nedges = edges.size(0)

        with torch.no_grad():
            #un_lpots = model.get_obs_lps(batch) # bsz x T x K log unary potentials
            ed_lpots = model.get_edge_scores(edges, T) # nedges x K*K log potentials

            exed_lpots = ed_lpots.view(nedges, 1, K, K)
            # get approximate unclamped marginals
            nodebs, facbs, _, _, _ = dolbp(exed_lpots, edges, niter=args.lbp_iter, renorm=True,
                                           randomize=args.randomize_lbp, tol=args.lbp_tol)

            tau_u = torch.stack([nodebs[t] for t in range(T)]).transpose(0, 1) # 1 x T x K
            tau_e = torch.stack(
                [facbs[e] for e in range(nedges)]).transpose(0, 1) # 1 x nedge x K x K
            tau_u, tau_e = (tau_u.exp() + EPS), (tau_e.exp() + EPS)

        for i in range(args.z_iter):
            with torch.no_grad(): # these functions are used in calc'ing the loss below too
                pred_rho = infnet.q(edges, T) # nedges x K^2 logits
                # should be 1 x T x K and 1 x nedges x K^2
                predtau_u, predtau_e, _ = get_taus_and_pens(pred_rho, nodeidxs, K, neginf,
                                                            penfunc=penfunc)
                predtau_u, predtau_e = predtau_u.exp() + EPS, predtau_e.exp() + EPS

            # i guess we'll just pick one entry from each
            un_margs = tau_u[0][:, 0] # T
            bin_margs = tau_e[0][:, K-1, K-1] # nedges
            pred_un_margs = predtau_u[0][:, 0] # T
            pred_bin_margs = predtau_e[0].view(nedges, K, K)[:, K-1, K-1] # nedges
            print(i, "unary corr: %.4f, binary corr: %.4f" %
                  (corr(un_margs, pred_un_margs),
                   corr(bin_margs, pred_bin_margs)))

            popt2.zero_grad()
            pred_rho = infnet.q(edges, T) # nedges x K^2 logits
            in_loss, ipen_loss = inner_lossz(pred_rho.view(1, -1), ed_lpots.view(1, -1), nodeidxs,
                                             K, ne, neginf, penfunc)
            in_loss = in_loss + args.pen_mult/npenterms * ipen_loss
            print("in_loss", in_loss.item())
            in_loss.backward()
            clip_opt_params(popt2, args.clip)
            popt2.step()
        exit()

    bad_epochs = -1
    for ep in range(args.epochs):
        if args.loss == "exact":
            ll, ntokes = exact_train(trbatches, model, popt1, helper, device, args)
            print("Epoch {:3d} | train tru-ppl {:8.3f}".format(
                ep, math.exp(-ll/ntokes)))
            with torch.no_grad():
                vll, vntokes = exact_validate(valbatches, model, helper, device)
                print("Epoch {:3d} | val tru-ppl {:8.3f}".format(
                    ep, math.exp(-vll/vntokes)))
                # if ep == 4 and math.exp(-vll/vntokes) >= 280:
                #     break
            voloss = -vll
        elif args.loss == "lbp":
            oloss, nex = lbp_train(trbatches, model, popt1, helper, device, args)
            print("Epoch {:3d} | train out_loss {:8.3f}".format(
                ep, oloss/nex))
            with torch.no_grad():
                voloss, vnex = lbp_validate(valbatches, model, helper, device)
                print("Epoch {:3d} | val out_loss {:8.3f} ".format(
                    ep, voloss/vnex))
        else: # infnet
            oloss, iloss, ploss, nex = train_unsup_am(trbatches, model, infnet, popt1, popt2, cache,
                                                      penfunc, neginf, device, args)
            print("Epoch {:3d} | train out_loss {:.3f} | train in_loss {:.3f}".format(
                ep, oloss/nex, iloss/nex))

            with torch.no_grad():
                voloss, vploss, vnex = validate_unsup_am(valbatches, model, infnet,
                                                         cache, penfunc, neginf, device, args)
                print("Epoch {:3d} | val out_loss {:.3f} | val barr_loss {:.3f}".format(
                    ep, voloss/vnex, vploss/vnex))

        if args.loss != "exact":
            with torch.no_grad():
                # trull, ntokes = exact_validate(trbatches, model, helper, device)
                # print("Epoch {:3d} | train tru-ppl {:.3f}".format(
                #     ep, math.exp(-trull/ntokes)))

                vll, vntokes = exact_validate(valbatches, model, helper, device)
                print("Epoch {:3d} | val tru-ppl {:.3f}".format(
                    ep, math.exp(-vll/vntokes)))
                voloss = -vll

            # trppl = math.exp(-trull/ntokes)
            # if (ep == 0 and  trppl > 3000) or (ep > 0 and trppl > 1000):
            #     break

        if voloss < best_loss:
            best_loss = voloss
            bad_epochs = -1
            print("updating best model")
            bestmodel.load_state_dict(model.state_dict())
            if bestinfnet is not None:
                bestinfnet.load_state_dict(infnet.state_dict())
            if len(args.save) > 0 and not args.grid:
                print("saving model to", args.save)
                torch.save({"opt": args, "mod_sd": bestmodel.state_dict(),
                            "inf_sd": bestinfnet.state_dict() if bestinfnet is not None else None,
                            "bestloss": bestloss}, args.save)
        if (voloss >= prev_loss or lrdecay) and args.optalg == "sgd":
            for group in popt1.param_groups:
                group['lr'] *= args.decay
            for group in popt2.param_groups:
                group['lr'] *= args.decay
            #decay = True
        if (voloss >= prev_loss or pendecay):
            args.pen_mult *= args.pendecay
            print("pen_mult now", args.pen_mult)
            pendecay = True

        prev_loss = voloss
        if ep >= 2 and math.exp(best_loss/vntokes) > 650:
            break
        print("")
        bad_epochs += 1
        if bad_epochs >= 5:
            break
        if args.reset_adam: #bad_epochs == 1:
            print("resetting adam...")
            for group in popt2.param_groups:
                group['lr'] *= args.decay # not really decay
        # if args.reset_adam and ep == 1: #bad_epochs == 1:
        #     print("resetting adam...")
        #     popt2 = torch.optim.Adam(infnet.parameters(), lr=args.ilr)

    return bestmodel, bestinfnet, best_loss



if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    corpus = data.SentCorpus(args.data, args.bsz, thresh=args.thresh,
                             max_len=args.max_len, vocab=None)
    helper = InfcHelper(args)

    print("total num batches", len(corpus.train))
    trbatches = corpus.train
    valbatches = corpus.valid

    max_seqlen = max(batch.size(0) for batch in corpus.train)
    max_verts = max_seqlen if args.use_length else args.markov_order

    def get_pms(opts):
        if opts.penfunc == "l2":
            pms = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
        else:
            pms = [0.1, 0.5, 1, 5, 10]
        return pms

    def get_vemb_size(opts):
        return [opts.lemb_size]

    hypers = OrderedDict({'optalg': ['adam'],
                          'pendecay': [1, 0.9, 0.7, 0.5],
                          'penfunc': ["l2", "js", "kl2", "kl1"],
                          'init': [0.05],
                          'qinit': [0.1],
                          'lr': [0.005, 0.003, 0.001, 0.0005, 0.0001], # got rid of 0.01
                          'ilr': [0.005, 0.003, 0.001, 0.0005], # got rid of 0.01
                          'lemb_size': [200],
                          'vemb_size': get_vemb_size,
                          'qemb_size': [100],
                          'q_hid_size': [64], #not used for transformer
                          'q_heads': [2],
                          'q_layers': [2],
                          'clip': [1, 5],
                          'pen_mult': [0.1, 0.5, 1, 5, 10],
                          #'seed': list(range(100000))
                          'seed': [70407],
                          'reset_adam': [True, False],
                          'z_iter': [1, 10, 20, 40, 50],
                          'zx_iter': [1, 10, 20, 40, 50],
                         })

    cache = {}

    if not args.grid:
        args.nruns = 1

    bestloss = float("inf")
    donez = set()
    for i in range(args.nruns):
        torch.manual_seed(i) # seed for this run just so we randomly get new choices
        if args.grid:
            while True:
                print("spinning!")
                hypkey = []
                for hyp, choices in hypers.items():
                    if isinstance(choices, list):
                        hypvals = choices
                    else: # it's a function
                        hypvals = choices(args)
                    choice = hypvals[torch.randint(len(hypvals), (1,)).item()]
                    args.__dict__[hyp] = choice
                    hypkey.append(choice)
                hypkey = tuple(hypkey)
                if hypkey not in donez:
                    donez.add(hypkey)
                    break

        bestmodel, bestinfnet, runloss = main(
            args, helper, cache, max_seqlen, max_verts, len(corpus.dictionary),
            trbatches, valbatches)
        if runloss < bestloss:
            bestloss = runloss
            if len(args.save) > 0:
                print("saving model to", args.save)
                torch.save({"opt": args, "mod_sd": bestmodel.state_dict(),
                            "inf_sd": bestinfnet.state_dict() if bestinfnet is not None else None,
                            "bestloss": bestloss}, args.save)
        print()
        print()
