import math
import argparse

from collections import OrderedDict

import random

import torch
import torch.nn as nn

#import data2 as data
from infc_utils import _multi_idx_loop as _multi_idx
from infc_utils import batch_fwdalg, batch_posterior_sample
from utils import clip_opt_params

class NeuralHMM(nn.Module):
    def __init__(self, ntypes, opt):
        super(NeuralHMM, self).__init__()
        self.K, self.M = opt.K, opt.markov_order
        self.drop = nn.Dropout(opt.dropout)

        self.lembs = nn.Parameter(torch.Tensor(opt.K, opt.lemb_size))
        self.tlembs = nn.Parameter(torch.Tensor(opt.K+1, opt.lemb_size))

        self.resid = not opt.not_residual

        if self.resid:
            self.decoder = nn.Linear(opt.lemb_size, ntypes)
            self.em_mlp = nn.Sequential(nn.Linear(opt.lemb_size, opt.lemb_size),
                                        nn.ReLU(), self.drop)
            self.em_norm = nn.LayerNorm(opt.lemb_size)
            self.trans_decoder = nn.Linear(self.M*opt.lemb_size, opt.K)
            self.trans_mlp = nn.Sequential(nn.Linear(self.M*opt.lemb_size, self.M*opt.lemb_size),
                                           nn.ReLU(), self.drop)
            self.trans_norm = nn.LayerNorm(self.M*opt.lemb_size)
        else:
            self.decoder = nn.Sequential(nn.Linear(opt.lemb_size, opt.wemb_size),
                                         nn.ReLU(), self.drop,
                                         nn.Linear(opt.wemb_size, ntypes))
            self.trans_decoder = nn.Sequential(nn.Linear(self.M*opt.lemb_size, opt.t_hid_size),
                                               nn.ReLU(), self.drop,
                                               nn.Linear(opt.t_hid_size, opt.K))

        self.pinit = opt.init
        self.init_weights()

    def init_weights(self):
        initrange = self.pinit
        if self.resid:
            lins = [self.decoder, self.em_mlp[0], self.trans_decoder, self.trans_mlp[0]]
        else:
            lins = [self.decoder[0], self.decoder[-1], self.trans_decoder[0],
                    self.trans_decoder[-1]]
        for lin in lins:
            lin.weight.data.uniform_(-initrange, initrange)
            if hasattr(lin, "bias"):
                lin.bias.data.zero_()
        params = [self.lembs, self.tlembs]
        for par in params:
            par.data.uniform_(-initrange, initrange)

    def init_word_embs(self, vocab, wrd2emb, freeze=False):
        dec = self.decoder if self.resid else self.decoder[-1]
        for i, wrd in enumerate(vocab):
            if wrd in wrd2emb:
                dec.weight.data[i].copy_(torch.from_numpy(wrd2emb[wrd]))
                #self.lut.weight.data[i].copy_(torch.from_numpy(wrd2emb[wrd]))
        if freeze:
            dec.weight.requires_grad = False
            #self.lut.weight.requires_grad = False

    def get_emdist(self):
        if self.resid:
            emdist = torch.log_softmax(
                self.decoder(self.em_norm(self.lembs + self.em_mlp(self.lembs))), dim=1) # K x V
        else:
            emdist = torch.log_softmax(self.decoder(self.lembs), dim=1) # K x V
        return emdist


    def get_transdist(self):
        cat_labe_embs = []
        Kp1 = self.K+1 # extra class for start state
        # make cartesian product of label embeddings
        for m in range(self.M):
            nreps = Kp1**(self.M-m-1) # number of times to repeat each label embedding
            # make block of size (K+1)^M-m x lemb_size
            block = self.tlembs.unsqueeze(1).repeat(1, nreps, 1).view(-1, self.tlembs.size(1))
            breps = Kp1**m # number of times to repeat each block
            cat_labe_embs.append(block.repeat(breps, 1))
        cat_labe_embs = torch.cat(cat_labe_embs, 1) # K+1^M x M*lemb_size
        assert cat_labe_embs.size(0) == Kp1**self.M
        if self.resid:
            tscores = self.trans_decoder( # K+1^M x K
                self.trans_norm(cat_labe_embs + self.trans_mlp(cat_labe_embs)))
        else:
            tscores = self.trans_decoder(cat_labe_embs) # K+1^M x K
        tdims = [Kp1]*self.M
        tdims.append(self.K)
        return torch.log_softmax(tscores, dim=1).view(*tdims)


    def log_joint(self, x, z, emdist, transdist):
        """
        emdist - K x V log normalized
        transdist - K+1 x K+1 x ... x K log normalized transition mat
        returns bsz-length vector
        """
        T, bsz = x.size()
        K, M = emdist.size(0), transdist.dim() - 1
        # get emission logprobs
        xlps = emdist.t()[x] # T x bsz x K
        emlps = xlps.view(-1, K).gather(1, z.view(-1, 1)).view(T, -1).sum(0) # T*bsz -> bsz

        # pad so we can easily index
        z = torch.cat([z.new(M, bsz).fill_(K), z], 0)

        # possibly easiest to loop
        trans_lps = [_multi_idx(transdist, z[t-M:t+1, b])
                     for t in range(M, T+M) for b in range(bsz)]

        lps = emlps + torch.stack(trans_lps).view(T, -1).sum(0)
        return lps


    def log_joint_ohz(self, x, z, z_oh, emdist, transdist):
        """
        z_oh is a T x bsz x K one-hot representation of z
        """
        T, bsz = x.size()
        K, M = emdist.size(0), transdist.dim() - 1
        # get emission logprobs
        xlps = emdist.t()[x] # T x bsz x K
        emlps = (xlps*z_oh).sum(2).sum(0) # bsz

        # pad
        z = torch.cat([z.new(M, bsz).fill_(K), z], 0)

        # possibly easiest to loop
        trans_lps = [(_multi_idx(transdist, z[t-M:t, b])*z_oh[t-M][b]).sum()
                     for t in range(M, T+M) for b in range(bsz)]

        lps = emlps + torch.stack(trans_lps).view(T, -1).sum(0)
        return lps



class MFHMMInfNet(nn.Module):
    def __init__(self, ntypes, opt):
        super(MFHMMInfNet, self).__init__()
        self.K, self.M = opt.K, opt.markov_order
        self.drop = nn.Dropout(opt.dropout)

        self.resinf = not opt.not_resinf
        if self.resinf:
            self.inf_norm = nn.LayerNorm(opt.qemb_size)
            self.brnn = nn.LSTM(opt.qemb_size, opt.qemb_size//2, num_layers=opt.qlayers,
                                bidirectional=True)
            self.inf_decoder = nn.Sequential(self.drop, nn.Linear(opt.qemb_size, opt.K))
        else:
            self.brnn = nn.LSTM(opt.qemb_size, opt.qhid_size, num_layers=opt.qlayers,
                                bidirectional=True)
            self.inf_decoder = nn.Sequential(self.drop, nn.Linear(2*opt.qhid_size, opt.K))

        self.lut = nn.Embedding(ntypes, opt.qemb_size)
        self.use_inpdep_bl = opt.use_inpdep_bl
        if opt.use_inpdep_bl:
            self.baseline_dec = nn.Sequential(nn.Linear(2*opt.qhid_size, opt.qhid_size),
                                              nn.ReLU(), self.drop,
                                              nn.Linear(opt.qhid_size, 1))

        self.pinit = opt.init
        self.init_weights()

    def init_weights(self):
        initrange = self.pinit
        lins = [self.inf_decoder[1], self.lut]
        if self.use_inpdep_bl:
            lins.extend([self.baseline_dec[0], self.baseline_dec[-1]])
        for lin in lins:
            lin.weight.data.uniform_(-initrange, initrange)
            if hasattr(lin, "bias"):
                lin.bias.data.zero_()
        for thing in self.brnn.parameters():
            thing.data.uniform_(-initrange, initrange)

    def q(self, x):
        """
        returns T x bsz x K approx posteriors at each timestep
        """
        T, bsz = x.size()
        emb = self.lut(x) # T x bsz x qemb_size
        #emb = self.drop(emb)
        output, (hT, _) = self.brnn(emb) # T x bsz x 2*hid_size, layers*2 x batch x hid_size
        output = self.drop(output)
        if self.resinf:
            decoded = torch.softmax(
                self.inf_decoder(
                    self.inf_norm(
                        emb.view(-1, emb.size(2)) + output.view(-1, output.size(2)))), dim=1)
        else:
            decoded = torch.softmax( # T*bsz x K
                self.inf_decoder(output.view(-1, output.size(2))), dim=1)



        # just doing this here for convenience
        if self.use_inpdep_bl:
            # note that hT.view(layers, 2, bsz, hid_size)[layer] contains final states
            # from either direction, so don't need to muck with it!
            finals = hT.transpose(0, 1).contiguous().view( # lay*2 x bsz x hid -> bsz x lay*2 x hid
                bsz, self.brnn.num_layers, 2*self.brnn.hidden_size).sum(1) # -> bsz x 2*hid
            baseline = self.baseline_dec(finals).squeeze(1) # bsz
        else:
            baseline = 0
        return decoded.view(T, bsz, -1), baseline


class FOHMMInfNet(nn.Module):
    def __init__(self, ntypes, opt):
        super(FOHMMInfNet, self).__init__()
        self.K, self.M = opt.K, 1
        self.drop = nn.Dropout(opt.dropout)

        self.lembs = nn.Parameter(torch.Tensor(opt.K, opt.lemb_size))
        self.tlembs = nn.Parameter(torch.Tensor(opt.K+1, opt.lemb_size))

        self.resid = True

        if self.resid:
            self.decoder = nn.Linear(opt.lemb_size + 2*opt.qhid_size, ntypes)
            self.em_mlp = nn.Sequential(nn.Linear(opt.lemb_size + 2*opt.qhid_size,
                                                  opt.lemb_size + 2*opt.qhid_size),
                                        nn.ReLU(), self.drop)
            self.em_norm = nn.LayerNorm(opt.lemb_size + 2*opt.qhid_size)
            self.trans_decoder = nn.Linear(self.M*opt.lemb_size + 2*opt.qhid_size, opt.K)
            self.trans_mlp = nn.Sequential(nn.Linear(self.M*opt.lemb_size + 2*opt.qhid_size,
                                                     self.M*opt.lemb_size + 2*opt.qhid_size),
                                           nn.ReLU(), self.drop)
            self.trans_norm = nn.LayerNorm(self.M*opt.lemb_size + 2*opt.qhid_size)
        else:
            self.decoder = nn.Sequential(nn.Linear(opt.lemb_size + 2*opt.qhid_size, opt.wemb_size),
                                         nn.ReLU(), self.drop,
                                         nn.Linear(opt.wemb_size, ntypes))
            self.trans_decoder = nn.Sequential(nn.Linear(self.M*opt.lemb_size + 2*opt.qhid_size,
                                                         opt.t_hid_size),
                                               nn.ReLU(), self.drop,
                                               nn.Linear(opt.t_hid_size, opt.K))

        self.brnn = nn.LSTM(opt.qemb_size, opt.qhid_size, num_layers=opt.qlayers,
                            bidirectional=True, dropout=opt.dropout)
        self.lut = nn.Embedding(ntypes, opt.qemb_size)
        self.use_inpdep_bl = opt.use_inpdep_bl
        if opt.use_inpdep_bl:
            self.baseline_dec = nn.Sequential(nn.Linear(2*opt.qhid_size, opt.qhid_size),
                                              nn.ReLU(), self.drop,
                                              nn.Linear(opt.qhid_size, 1))
        self.pinit = opt.init
        self.init_weights()

    def init_weights(self):
        initrange = self.pinit
        if self.resid:
            lins = [self.decoder, self.em_mlp[0], self.trans_decoder, self.trans_mlp[0]]
        else:
            lins = [self.decoder[0], self.decoder[-1], self.trans_decoder[0],
                    self.trans_decoder[-1]]
        lins.append(self.lut)
        if self.use_inpdep_bl:
            lins.extend([self.baseline_dec[0], self.baseline_dec[-1]])
        for lin in lins:
            lin.weight.data.uniform_(-initrange, initrange)
            if hasattr(lin, "bias"):
                lin.bias.data.zero_()

        params = [self.lembs, self.tlembs]
        for par in params:
            par.data.uniform_(-initrange, initrange)

        for thing in self.brnn.parameters():
            thing.data.uniform_(-initrange, initrange)

    def get_emdist(self, xenc):
        """
        xenc - bsz x dim
        returns bsz x K x V log normalized
        """
        bsz = xenc.size(0)
        inp = torch.cat([self.lembs.repeat(bsz, 1), # bsz*K x embsz+enc_size
                         xenc.unsqueeze(1).repeat(1, self.K, 1).view(bsz*self.K, -1)], 1)
        if self.resid:
            emdist = torch.log_softmax(
                self.decoder(self.em_norm(inp + self.em_mlp(inp))), dim=1) # bsz*K x V
        else:
            emdist = torch.log_softmax(self.decoder(inp), dim=1) # bsz*K x V
        return emdist.view(bsz, self.K, -1) # bsz x K x V

    def get_transdist(self, xenc):
        """
        xenc - bsz x dim
        returns bsz x K+1 x K log normalized
        """
        bsz = xenc.size(0)
        cat_labe_embs = []
        Kp1 = self.K+1 # extra class for start state
        # make cartesian product of label embeddings
        for m in range(self.M):
            nreps = Kp1**(self.M-m-1) # number of times to repeat each label embedding
            # make block of size (K+1)^M-m x lemb_size
            block = self.tlembs.unsqueeze(1).repeat(1, nreps, 1).view(-1, self.tlembs.size(1))
            breps = Kp1**m # number of times to repeat each block
            cat_labe_embs.append(block.repeat(breps, 1))
        cat_labe_embs = torch.cat(cat_labe_embs, 1) # K+1^M x M*lemb_size
        assert cat_labe_embs.size(0) == Kp1**self.M # for inf M is always 1
        cat_labe_embs = torch.cat([cat_labe_embs.repeat(bsz, 1), # Kp1 x lemb_size + enc_size
                                   xenc.unsqueeze(1).repeat(1, Kp1, 1).view(bsz*Kp1, -1)], 1)
        if self.resid:
            tscores = self.trans_decoder( # bsz*K+1^M x K
                self.trans_norm(cat_labe_embs + self.trans_mlp(cat_labe_embs)))
        else:
            tscores = self.trans_decoder(cat_labe_embs) # bsz*K+1^M x K
        tdims = [bsz]
        tdims.extend([Kp1]*self.M)
        tdims.append(self.K)
        return torch.log_softmax(tscores, dim=1).view(*tdims) # bsz x K+1 x K

    def q(self, x):
        """
        returns T x bsz x K approx posteriors at each timestep
        """
        T, bsz = x.size()
        emb = self.lut(x) # T x bsz x qemb_size
        #emb = self.drop(emb)
        output, (hT, _) = self.brnn(emb) # T x bsz x 2*hid_size, layers*2 x batch x hid_size
        #output = self.drop(output)
        xenc = output.mean(0) # bsz x 2*hid_size
        emdist = self.get_emdist(xenc) # bsz x K x V
        transdist = self.get_transdist(xenc) # bsz x K+1 x K

        # just doing this here for convenience
        if self.use_inpdep_bl:
            # note that hT.view(layers, 2, bsz, hid_size)[layer] contains final states
            # from either direction, so don't need to muck with it!
            finals = hT.transpose(0, 1).contiguous().view( # lay*2 x bsz x hid -> bsz x lay*2 x hid
                bsz, self.brnn.num_layers, 2*self.brnn.hidden_size).sum(1) # -> bsz x 2*hid
            baseline = self.baseline_dec(finals).squeeze() # bsz
        else:
            baseline = 0
        return emdist, transdist, baseline



def to_one_hot(idxs, K):
    """
    idxs - T x bsz
    """
    oh = torch.zeros(idxs.size(0), idxs.size(1), K).to(idxs.device)
    oh.view(-1, K).scatter_(1, idxs.view(-1, 1), 1)
    return oh


def reinforce_elbo(model, infnet, x, curr_mean, curr_var, alph=0.8, nsamps=1, vimco=False):
    """
    posteriors - T x bsz x K
    """
    T, bsz = x.size()
    if isinstance(infnet, FOHMMInfNet): # first order HMM approx posterior
        qemdist, qtransdist, inp_dep_bl = infnet.q(x)
        z, ln_qz = batch_posterior_sample(x, qemdist, qtransdist, nsamps=nsamps)
    else:
        # get approximate mean-field posteriors
        posteriors, inp_dep_bl = infnet.q(x) # T x bsz x K, bsz
        K = posteriors.size(2)
        # sample: T*bsz x nsamps.
        z = torch.multinomial(posteriors.view(-1, K), num_samples=nsamps, replacement=True)
        assert not z.requires_grad
        #  T*bsz x nsamps -> T x bsz*nsamps -> bsz*nsamps
        ln_qz = posteriors.view(-1, K).gather(1, z.view(-1, nsamps)).view(T, -1).sum(0)

    # we use the "surrogate loss":
    # E_q [ (ln p(x, z) - ln q(z)).detach() ln q(z) + (ln p(x, z) - ln q(z))],
    # which has the gradients we want
    emdist, transdist = model.get_emdist(), model.get_transdist()
    if nsamps > 1:
        repx = x.view(T, bsz, 1).repeat(1, 1, nsamps).view(T, -1) # T x bsz*nsamps
        ln_pxz = model.log_joint(repx, z.view(T, -1), emdist, transdist) # bsz*nsamps
        sampsigs = ln_pxz.view(bsz, -1) - ln_qz.view(bsz, -1) # bsz x nsamps
        signal = torch.logsumexp(sampsigs, dim=1).squeeze() - math.log(nsamps) # bsz
        if vimco:
            dsampsigs = sampsigs.detach().clone() # yells at me if i don't also clone
            sampbls = []
            sigsums = dsampsigs.sum(1) # bsz
            for k in range(nsamps):
                temp = dsampsigs[:, k]
                dsampsigs[:, k] = (sigsums - temp)/(nsamps-1)
                sampbls.append(torch.logsumexp(dsampsigs, dim=1) - math.log(nsamps))
                dsampsigs[:, k] = temp
            sampbls = torch.stack(sampbls).t() # bsz x nsamps
        else:
            sampbls = 0
    else:
        ln_pxz = model.log_joint(x, z.view(T, bsz), emdist, transdist) # bsz
        signal = (ln_pxz - ln_qz) # bsz
    # the baselines below follow the NVIL paper
    detached_signal = (signal - inp_dep_bl).detach() # bsz
    curr_mean = alph*curr_mean + (1-alph)*detached_signal.mean()
    curr_var = alph*curr_var + (1-alph)*detached_signal.var()
    detached_signal.add_(-curr_mean).div_(max(1, curr_var.sqrt().item())) # bsz
    # form the surrogate loss with our baselines
    if nsamps > 1: # just scale by the signal and add
        surr_elbo = ((detached_signal.view(bsz, 1) - sampbls)
                     * ln_qz.view(bsz, -1)).sum(1) + signal # bsz
    else:
        surr_elbo = detached_signal * ln_qz + signal # bsz

    # descend in negative elbo
    if infnet.use_inpdep_bl and not vimco: # no baselines for vimco for now
        #print("oy", inp_dep_bl.size(), detached_signal.size())
        mseloss = torch.nn.functional.mse_loss(inp_dep_bl, detached_signal, reduction='sum')
        (mseloss - surr_elbo.sum()).div(bsz).backward()
    else:
        surr_elbo.sum().div(-bsz).backward() # negate etc

    # signal should contain ELBO or IWAE bound for each thing in the batch
    return signal.sum().item(), curr_mean, curr_var


def gumbel_st_elbo(model, infnet, x, eps=1e-20):
    T, bsz = x.size()
    nsamps = 1
    # get approximate posteriors
    posteriors, _ = infnet.q(x) # T x bsz x K, bsz
    K = posteriors.size(2)
    # add gumbel noise: g = -log(-log(u)) if u is from uniform[0, 1]
    perturbed = posteriors - torch.log(-torch.log(torch.rand_like(posteriors) + eps))
    _, z = perturbed.view(-1, K).max(1)
    # turn to one-hot so we can automatically get grads
    z_oh = to_one_hot(z.view(T, bsz), K) # T x bsz x K
    z_oh.requires_grad = True
    #ln_qz = posteriors.view(-1, K).gather(1, z.view(-1, nsamps)).view(T, -1).sum(0)
    ln_qz = (posteriors*z_oh).sum(2).sum(0) # bsz
    emdist, transdist = model.get_emdist(), model.get_transdist()
    ln_pxz = model.log_joint_ohz(x, z.view(T, bsz), z_oh, emdist, transdist) # bsz
    signal = (ln_pxz - ln_qz) # bsz
    belbo = signal.sum()
    # descent in negative elbo
    belbo.div(-bsz).backward(retain_graph=True) # negate etc
    # pass grads wrt z straight through
    perturbed.backward(z_oh.grad)
    return belbo.item()


def val_elbo(model, infnet, x, emdist, transdist, nsamps=1):
    model.eval()
    infnet.eval()
    T, bsz = x.size()
    # get approximate posteriors
    if isinstance(infnet, FOHMMInfNet):
        qemdist, qtransdist, _ = infnet.q(x)
        z, ln_qz = batch_posterior_sample(x, qemdist, qtransdist, nsamps=nsamps)
    else:
        posteriors, _ = infnet.q(x) # T x bsz x K, bsz
        K = posteriors.size(2)
        # sample: T*bsz x nsamps.
        z = torch.multinomial(posteriors.view(-1, K), num_samples=nsamps, replacement=True)
        #  T*bsz x nsamps -> T x bsz*nsamps -> bsz*nsamps
        ln_qz = posteriors.view(-1, K).gather(1, z.view(-1, nsamps)).view(T, -1).sum(0)
    if nsamps > 1:
        repx = x.view(T, bsz, 1).repeat(1, 1, nsamps).view(T, -1) # T x bsz*nsamps
        ln_pxz = model.log_joint(repx, z.view(T, -1), emdist, transdist) # bsz*nsamps
        sampsigs = ln_pxz.view(bsz, -1) - ln_qz.view(bsz, -1) # bsz x nsamps
        signal = torch.logsumexp(sampsigs, dim=1).squeeze().add_(-math.log(nsamps)) # bsz
    else:
        ln_pxz = model.log_joint(x, z.view(T, bsz), emdist, transdist) # bsz
        signal = (ln_pxz - ln_qz) # bsz
    # signal should contain ELBO or IWAE bound for each thing in the batch
    return signal.sum().item()


def train(corpus, model, infnet, optim, args, device):
    model.train()
    infnet.train()
    K = args.K
    elbo, ntokens = 0.0, 0
    perm = torch.randperm(len(corpus))
    mean, var = None, None
    for i, idx in enumerate(perm):
        optim.zero_grad()
        batch = corpus[idx.item()].to(device)
        if args.reinforce:
            if mean is None:
                mean, var, balph = 0, 0, 0
            else:
                balph = args.alpha
            belbo, mean, var = reinforce_elbo(
                model, infnet, batch, mean, var, alph=balph, nsamps=args.nsamps,
                vimco=args.vimco)
        else: # gumbel + st
            belbo = gumbel_st_elbo(model, infnet, batch)
        elbo += belbo
        ntokens += batch.nelement()
        clip_opt_params(optim, args.clip)
        optim.step()
        if (i+1) % args.log_interval == 0:
            print("{:5d}/{:5d} | lr {:02.4f} | ppl {:8.2f}".format(
                i+1, perm.size(0), args.lr, math.exp(-elbo/ntokens)))
    return elbo, ntokens


def exact_train(corpus, model, optim, args, device):
    """
    this is exact
    """
    model.train()
    K = args.K
    elbo, ntokens = 0.0, 0
    perm = torch.randperm(len(corpus))
    #exact_logmarg = batch_fwdalg if args.markov_order == 1 else batch_var_elim
    exact_logmarg = batch_fwdalg
    for i, idx in enumerate(perm):
        # if i > 1:
        #     break
        optim.zero_grad()
        batch = corpus[idx.item()].to(device)
        emdist, transdist = model.get_emdist(), model.get_transdist()
        btrull = exact_logmarg(batch, emdist, transdist).sum()
        belbo = btrull.item()
        btrull.div(-batch.size(1)).backward()
        elbo += belbo
        ntokens += batch.nelement()
        clip_opt_params(optim, args.clip)
        optim.step()
        if (i+1) % args.log_interval == 0:
            print("{:5d}/{:5d} | lr {:02.4f} | ppl {:8.2f}".format(
                i+1, perm.size(0), args.lr, math.exp(-elbo/ntokens)))
    return elbo, ntokens



def validate(corpus, model, infnet, args, device, just_exact=False):
    model.eval()
    K = args.K
    elbo, trull, ntokens = 0.0, 0.0, 0
    #exact_logmarg = batch_fwdalg if args.markov_order == 1 else batch_var_elim
    exact_logmarg = batch_fwdalg
    for i in range(len(corpus)):
        batch = corpus[i].to(device)
        emdist, transdist = model.get_emdist(), model.get_transdist()
        if just_exact:
            belbo = 0
        else:
            belbo = val_elbo(model, infnet, batch, emdist, transdist, nsamps=args.nval_samps)
        btrull = exact_logmarg(batch, emdist, transdist).sum().item()
        elbo += belbo
        trull += btrull
        ntokens += batch.nelement()
    return elbo, trull, ntokens


parser = argparse.ArgumentParser(description='')
parser.add_argument('-data', type=str, default="/scratch/data/ptb/",
                    #default="/scratch/data/PTB/ptb_processed/dependency/",
                    help='location of the data corpus')
parser.add_argument('-wvec_path', type=str, default="/scratch/code/struct-learning-with-flow/struct_flow_data/wsj_word_vec.pc",
                    help='')
parser.add_argument('-freeze', action='store_true', help='freeze word embeddings')
parser.add_argument('-use_pt_wembs', action='store_true', help='freeze word embeddings')
parser.add_argument('-thresh', type=int, default=0, help='')
parser.add_argument('-max_len', type=int, default=20, help='')

parser.add_argument('-not_residual', action='store_true', help='')
parser.add_argument('-not_resinf', action='store_true', help='')
parser.add_argument('-wemb_size', type=int, default=100,
                    help='size of word embeddings [not used if residual=True]')
parser.add_argument('-lemb_size', type=int, default=100, help='size of latent label embeddings')
parser.add_argument('-t_hid_size', type=int, default=100,
                    help='transition hid size [not used if residual=True]')
parser.add_argument('-K', type=int, default=12, help='')
parser.add_argument('-markov_order', type=int, default=1, help='')

parser.add_argument('-qemb_size', type=int, default=100, help='size of infc embeddings')
parser.add_argument('-qhid_size', type=int, default=100, help='size of infc embeddings')
parser.add_argument('-qlayers', type=int, default=2, help='size of infc embeddings')

parser.add_argument('-exact', action='store_true', help='')
parser.add_argument('-reinforce', action='store_true', help='')
parser.add_argument('-vimco', action='store_true', help='')
parser.add_argument('-first_order_q', action='store_true', help='')
parser.add_argument('-alpha', type=float, default=0.0, help='baseline thing')
parser.add_argument('-use_inpdep_bl', action='store_true', help='')
parser.add_argument('-nval_samps', type=int, default=1, help='')
parser.add_argument('-nsamps', type=int, default=1, help='')

#parser.add_argument('-val_iter', type=int, default=100, help='')
parser.add_argument('-valbatches', type=int, default=0, help='')

parser.add_argument('-optalg', type=str, default='sgd',
                    choices=['sgd', 'adagrad', 'adam'], help='')
parser.add_argument('-init', type=float, default=0.1, help='param init')
parser.add_argument('-lr', type=float, default=1, help='initial learning rate')
parser.add_argument('-ilr', type=float, default=1, help='initial learning rate')
parser.add_argument('-decay', type=float, default=0.5, help='initial learning rate')
parser.add_argument('-clip', type=float, default=5, help='gradient clipping')
parser.add_argument('-epochs', type=int, default=40, help='upper epoch limit')
parser.add_argument('-bsz', type=int, default=16, help='batch size')
parser.add_argument('-dropout', type=float, default=0.2, help='dropout')
parser.add_argument('-seed', type=int, default=1111, help='random seed')
parser.add_argument('-cuda', action='store_true', help='use CUDA')

parser.add_argument('-log_interval', type=int, default=200, help='report interval')
parser.add_argument('-save', type=str, default='', help='path to save the final model')
parser.add_argument('-train_from', type=str, default='', help='')
parser.add_argument('-nruns', type=int, default=100, help='random seed')
parser.add_argument('-no_grid', action='store_true', help='')

def main(args, ntypes, trbatches, valbatches):
    print("main args", args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    model = NeuralHMM(ntypes, args).to(device)
    # if args.use_pt_wembs:
    #     model.init_word_embs(corpus.dictionary.idx2word, word_embs, freeze=args.freeze)
    if args.first_order_q:
        infnet = FOHMMInfNet(ntypes, args).to(device)
    else:
        infnet = MFHMMInfNet(ntypes, args).to(device)

    bestmodel = NeuralHMM(ntypes, args)
    if args.exact:
        bestinfnet = None
    else:
        bestinfnet = FOHMMInfNet(ntypes, args) if args.first_order_q else MFHMMInfNet(ntypes, args)

    best_loss, prev_loss = float("inf"), float("inf")
    decay = False
    if args.optalg == "sgd":
        optim = torch.optim.SGD(
            [{"params": model.parameters(), "lr": args.lr},
             {"params": infnet.parameters(), "lr": args.ilr}])
    elif args.optalg == "adagrad":
        optim = torch.optim.Adagrad(
            [{"params": model.parameters(), "lr": args.lr},
             {"params": infnet.parameters(), "lr": args.ilr}],
            initial_accumulator_value=0.1)
    else:
        optim = torch.optim.Adam(
            [{"params": model.parameters(), "lr": args.lr},
             {"params": infnet.parameters(), "lr": args.ilr}])

    bad_epochs = -1
    for ep in range(args.epochs):
        if args.exact:
            elbo, ntokes = exact_train(trbatches, model, optim, args, device)
        else:
            elbo, ntokes = train(trbatches, model, infnet, optim, args, device)
        print("Epoch {:3d} | train elbo-ppl {:8.3f}".format(
            ep, math.exp(-elbo/ntokes)))
        # if not args.exact:
        #     with torch.no_grad():
        #         _, trull, ntokes = validate(trbatches, model, infnet, args, device, just_exact=True)
        #     print("Epoch {:3d} | train tru-ppl {:8.3f}".format(
        #         ep, math.exp(-trull/ntokes)))
        with torch.no_grad():
            velbo, vtrull, vntokes = validate(valbatches, model, infnet, args, device)
        print("Epoch {:3d} | val elbo-ppl {:8.3f} | val tru-ppl {:8.3f}".format(
            ep, math.exp(-velbo/vntokes), math.exp(-vtrull/vntokes)))
        # if math.exp(-vtrull/vntokes) > 380:
        #     break
        print("")
        #voloss = -velbo if not args.exact else -vtrull
        voloss = -vtrull
        # pick a random subset to evaluate on
        if voloss < best_loss:
            best_loss = voloss
            bad_epochs = -1
            print("updating best model")
            bestmodel.load_state_dict(model.state_dict())
            if bestinfnet is not None:
                bestinfnet.load_state_dict(infnet.state_dict())
            # if len(args.save) > 0:
            #     print("saving model to", args.save)
            #     torch.save({"opt": args, "sd": model.state_dict(),
            #                 "bestloss": best_loss}, args.save)
        if (voloss >= prev_loss or decay) and args.optalg == "sgd":
            args.lr *= args.decay
            for group in optim.param_groups:
                group['lr'] = args.lr
            #decay = True
        prev_loss = voloss
        if ep >= 2 and math.exp(best_loss/vntokes) > 700:
            break
        # if args.lr <= 1e-5:
        #     break
        bad_epochs += 1
        if bad_epochs >= 5:
            break
        print("")
    return bestmodel, bestinfnet, best_loss


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    assert not args.train_from

    if args.use_pt_wembs:
        with open(args.wvec_path, 'rb') as f:
            word_embs = pickle.load(f) # word -> np array
            vocab = sorted(word_embs.keys())
    else:
        vocab = None

    # corpus = data.UTagCorpus(args.data, args.bsz, thresh=args.thresh,
    #                          max_len=args.max_len, vocab=vocab)

    import data
    corpus = data.SentCorpus(args.data, args.bsz, thresh=args.thresh,
                              max_len=args.max_len, vocab=None)
    trbatches = corpus.train
    valbatches = corpus.valid

    # print("total num batches", len(corpus.train))
    # if args.valbatches > 0:
    #     # redo seed so everyone gets the same batches
    #     torch.manual_seed(1111)
    #     perm = torch.randperm(len(corpus.train))
    #     valbatches = [corpus.train[idx.item()] for idx in perm[:args.valbatches]]
    #     trbatches = [corpus.train[idx.item()] for idx in perm[args.valbatches:]]
    #     print("now {:5d} trbatches and {:5d} valbatches".format(len(trbatches), len(valbatches)))
    #     print("valsig:", perm[:args.valbatches].sum().item())
    # else:
    #     trbatches = corpus.train
    #     valbatches = trbatches


    print("doing grid search stuff")
    # algs = ["adam"]
    # lrs = {"sgd": [1, 0.5, 0.1, 0.05, 0.01],
    #        "adagrad": [0.5, 0.3, 0.1, 0.03, 0.01],
    #        "adam": [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]}
    # lembs = [64, 100, 200]
    # clips = [1, 5]
    # qembs = [64, 100, 200]
    # lays = [1, 2, 3, 4]
    # inits = [0.1, 0.05, 0.01, 0.005]
    # qinits = [0.1, 0.05, 0.01, 0.005]



    hypers = OrderedDict({'optalg': ['adam'],
                          'init': [0.1, 0.05, 0.01, 0.005, 0.001],
                          'qinit': [0.1, 0.05, 0.01, 0.005, 0.001],
                          'lr': [0.003, 0.001, 0.0003, 0.0001, 0.00003], # got rid of 0.01
                          'ilr': [0.003, 0.001, 0.0003, 0.0001, 0.00003], # got rid of 0.01
                          'lemb_size': [64, 100, 200],
                          'qemb_size': [64, 100, 150, 200], #[32, 50, 64, 100],
                          'q_hid_size': [64, 100, 200], #, 300],
                          'qlayers': [1, 2, 3, 4],
                          'clip': [1, 5],
                          'seed': list(range(100000)),
                          'use_inpdep_bl': [True, False],
                          'alpha': [0, 0.5, 0.7, 0.8, 0.9, 1],
                         })


    torch.manual_seed(args.seed)

    if args.no_grid:
        args.nruns = 1

    bestloss = float("inf")
    for _ in range(args.nruns):
        if not args.no_grid:
            for hyp, choices in hypers.items():
                if isinstance(choices, list):
                    hypvals = choices
                else: # it's a function
                    hypvals = choices(args)
                choice = hypvals[torch.randint(len(hypvals), (1,)).item()]
                args.__dict__[hyp] = choice

        bestmodel, bestinfnet, runloss = main(
            args, len(corpus.dictionary), trbatches, valbatches)
        if runloss < bestloss:
            bestloss = runloss
            if len(args.save) > 0:
                print("saving model to", args.save)
                torch.save({"opt": args, "mod_sd": bestmodel.state_dict(),
                            "inf_sd": bestinfnet.state_dict() if bestinfnet is not None else None,
                            "bestloss": bestloss}, args.save)
        print()
        print()
