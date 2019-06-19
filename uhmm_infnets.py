import torch
import torch.nn as nn

from model_utils import ResidualLayer
from moarmodels import Encoder as TransformerEncoder

class RNodeInfNet(nn.Module):
    def __init__(self, ntypes, max_verts, opt):
        super(RNodeInfNet, self).__init__()
        self.K = opt.K
        self.drop = nn.Dropout(opt.dropout)
        self.inf_resid = not opt.not_inf_residual
        qemb_sz = opt.qemb_size
        self.ilut = nn.Embedding(max_verts+1, qemb_sz)
        self.wlut = nn.Embedding(ntypes, qemb_sz)

        q_in_size = 2*qemb_sz
        qx_in_size = 3*qemb_sz
        q_hid_size = opt.q_hid_size
        qx_hid_size = opt.q_hid_size


        if self.inf_resid:
            self.q_rnn = nn.ModuleList()
            self.qx_rnn = nn.ModuleList()
            self.q_rnn.append(nn.LSTM(q_in_size, q_in_size//2, num_layers=1,
                                      bidirectional=True))
            self.qx_rnn.append(nn.LSTM(qx_in_size, qx_in_size//2, num_layers=1,
                                       bidirectional=True))
            for _ in range(1, opt.q_layers):
                self.q_rnn.append(nn.LSTM(q_in_size, q_in_size//2, num_layers=1,
                                          bidirectional=True))
                self.qx_rnn.append(nn.LSTM(qx_in_size, qx_in_size//2, num_layers=1,
                                           bidirectional=True))
            self.q_decoder = nn.Sequential(ResidualLayer(q_in_size*2, q_in_size*2),
                                           ResidualLayer(q_in_size*2, q_in_size*2),
                                           self.drop,
                                           nn.Linear(q_in_size*2, self.K*self.K))
            self.qx_decoder = nn.Sequential(ResidualLayer(qx_in_size*2, qx_in_size*2),
                                            ResidualLayer(qx_in_size*2, qx_in_size*2),
                                            self.drop,
                                            nn.Linear(qx_in_size*2, self.K*self.K))
        else:
            self.q_rnn = nn.LSTM(q_in_size, q_hid_size, num_layers=opt.q_layers,
                                 bidirectional=True, dropout=opt.dropout)
            self.qx_rnn = nn.LSTM(qx_in_size, qx_hid_size, num_layers=opt.q_layers,
                                  bidirectional=True, dropout=opt.dropout)
            self.q_decoder = nn.Sequential(ResidualLayer(q_hid_size*4, q_hid_size*4),
                                           ResidualLayer(q_hid_size*4, q_hid_size*4),
                                           self.drop,
                                           nn.Linear(q_hid_size*4, self.K*self.K))
            self.qx_decoder = nn.Sequential(ResidualLayer(qx_hid_size*4, qx_hid_size*4),
                                            ResidualLayer(qx_hid_size*4, qx_hid_size*4),
                                            self.drop,
                                            nn.Linear(qx_hid_size*4, self.K*self.K))

        self.pinit = opt.qinit
        self.init_weights()

    def init_weights(self):
        initrange = self.pinit

        lins = [self.ilut, self.wlut, self.q_decoder[-1], self.qx_decoder[-1]]
        for lin in lins:
            lin.weight.data.uniform_(-initrange, initrange)
            if hasattr(lin, "bias"):
                lin.bias.data.zero_()

        if self.inf_resid:
            rnns = [rnn for rnn in self.q_rnn]
            rnns.extend([rnn for rnn in self.qx_rnn])
        else:
            rnns = [self.q_rnn, self.qx_rnn]
        for rnn in rnns:
            for thing in rnn.parameters():
                thing.data.uniform_(-initrange, initrange)

    def q(self, edges, T):
        """
        edges - nedges x 2
        returns 1 x nedges*K*K
        """
        nedges = edges.size(0)
        nodeembs = torch.cat([self.ilut.weight[:T],
                              self.ilut.weight[T].view(1, -1).expand(T, -1)], 1)
        inp = nodeembs.unsqueeze(1) # T x 1 x 2*embsize
        # maybe cat on a dummy thing if we use same rnn for infc
        if self.inf_resid:
            states, _ = self.q_rnn[0](inp) # T x 1 x rnn_size
            inp = self.drop(inp + states)
            for l in range(1, len(self.q_rnn)):
                states, _ = self.q_rnn[l](inp) # T x 1 x rnn_size
                inp = self.drop(states + inp)
            states = inp
        else:
            states, _ = self.q_rnn(inp) # T x 1 x rnn_size
        edgereps = states.squeeze(1)[edges] # nedges x 2 x qsize
        pseudo_lms = self.q_decoder(edgereps.view(nedges, -1)).view(-1) # nedges*K*K
        return pseudo_lms.view(1, -1)


    def qx(self, x, edges, T):
        """
        x - T x bsz
        edges - nedges x 2, where each entry is a 0-indexed symbol
        returns bsz x nedges*K*K
        """
        nedges = edges.size(0)
        bsz = x.size(1)
        wembs = self.wlut(x) # T x bsz x embsize
        nodeembs = torch.cat([self.ilut.weight[:T].view(T, 1, -1).expand(T, bsz, -1),
                              self.ilut.weight[T].view(1, 1, -1).expand(T, bsz, -1),
                              wembs], 2)
        inp = nodeembs # T x bsz x 3*embsize

        if self.inf_resid:
            states, _ = self.qx_rnn[0](inp) # T x bsz x rnn_size
            inp = self.drop(inp + states)
            for l in range(1, len(self.q_rnn)):
                states, _ = self.qx_rnn[l](inp) # T x bsz x rnn_size
                inp = self.drop(states + inp)
            states = inp
        else:
            states, _ = self.qx_rnn(inp) # T x bsz x rnn_size

        edgereps = states[edges] # nedges x 2 x bsz x rnnsize
        edgereps = edgereps.transpose(1, 2).transpose(0, 1) # bsz x nedges x 2 x qsize
        pseudo_lms = self.qx_decoder(edgereps.contiguous().view(bsz*nedges, -1))
        return pseudo_lms.view(bsz, -1)


class TNodeInfNet(nn.Module):
    def __init__(self, ntypes, max_verts, opt):
        super(TNodeInfNet, self).__init__()
        self.K = opt.K
        self.drop = nn.Dropout(opt.dropout)
        qemb_sz = opt.qemb_size
        self.ilut = nn.Embedding(max_verts+1, qemb_sz)
        self.wlut = nn.Embedding(ntypes, qemb_sz)

        q_in_size = 2*qemb_sz
        qx_in_size = 3*qemb_sz
        q_hid_size = q_in_size
        qx_hid_size = qx_in_size

        # fixing the key value dim to be 64????
        self.qt = TransformerEncoder(opt.q_layers, opt.q_heads, q_hid_size, 64, 64,
                                     opt.dropout, opt.dropout, 9999999)
        self.qxt = TransformerEncoder(opt.q_layers, opt.q_heads, qx_hid_size, 64, 64,
                                      opt.dropout, opt.dropout, 9999999)
        self.q_decoder = nn.Sequential(ResidualLayer(q_hid_size*2, q_hid_size*2),
                                       ResidualLayer(q_hid_size*2, q_hid_size*2),
                                       self.drop,
                                       nn.Linear(q_hid_size*2, self.K*self.K))
        self.qx_decoder = nn.Sequential(ResidualLayer(qx_hid_size*2, qx_hid_size*2),
                                        ResidualLayer(qx_hid_size*2, qx_hid_size*2),
                                        self.drop,
                                        nn.Linear(qx_hid_size*2, self.K*self.K))

        self.pinit = opt.qinit
        self.init_weights()

    def init_weights(self): # yoon seems to think default init is alright
        initrange = self.pinit

        lins = [self.ilut, self.wlut]#, self.q_decoder, self.qx_decoder]
        for lin in lins:
            lin.weight.data.uniform_(-initrange, initrange)
            if hasattr(lin, "bias"):
                lin.bias.data.zero_()

    def q(self, edges, T):
        """
        edges - nedges x 2
        returns 1 x nedges*K*K
        """
        nedges = edges.size(0)
        nodeembs = torch.cat([self.ilut.weight[:T],
                              self.ilut.weight[T].view(1, -1).expand(T, -1)], 1)
        inp = nodeembs.unsqueeze(0) # 1 x T x 2*embsize

        inp = self.qt(inp)

        # get edge reps
        edgereps = inp.squeeze(0)[edges] #nedges x 2 x qsize
        pseudo_lms = self.q_decoder(edgereps.view(nedges, -1)).view(-1) # nedges*K*K
        return pseudo_lms.view(1, -1)

    def qx(self, x, edges, T):
        """
        x - T x bsz
        edges - nedges x 2, where each entry is a 0-indexed symbol
        returns bsz x nedges*K*K
        """
        nedges = edges.size(0)
        bsz = x.size(1)
        wembs = self.wlut(x) # T x bsz x embsize
        nodeembs = torch.cat([self.ilut.weight[:T].view(T, 1, -1).expand(T, bsz, -1),
                              self.ilut.weight[T].view(1, 1, -1).expand(T, bsz, -1),
                              wembs], 2)
        inp = nodeembs.transpose(0, 1) # bsz x T x 3*embsize
        inp = self.qxt(inp)

        edgereps = inp.transpose(0, 1)[edges] # nedges x 2 x bsz x qsize
        edgereps = edgereps.transpose(1, 2).transpose(0, 1) # bsz x nedges x 2 x qsize
        pseudo_lms = self.qx_decoder(edgereps.contiguous().view(bsz*nedges, -1))
        return pseudo_lms.view(bsz, -1)
