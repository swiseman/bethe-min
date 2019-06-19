import torch
import torch.nn as nn

from model_utils import ResBlock, ResidualLayer
from moarmodels import Encoder as TransformerEncoder

class CNNEnc(nn.Module):
    def __init__(self, opt):
        super(CNNEnc, self).__init__()
        self.modlist = nn.ModuleList([ResBlock(2**i, opt) for i in range(opt.q_layers)])

    def forward(self, x):
        """
        x - T x bsz x dim
        returns T x bsz x dim
        """
        inp = x.transpose(0, 1).transpose(1, 2) # bsz x dim x T
        for l in range(len(self.modlist)):
            inp = self.modlist[l](inp)
        return inp.transpose(1, 2).transpose(0, 1)

class RNNEnc(nn.Module):
    def __init__(self, opt):
        super(RNNEnc, self).__init__()
        if "justvis" in opt.infarch or "dbl" in opt.infarch:
            insize = opt.qemb_sz
        else:
            insize = 2*opt.qemb_sz
        self.rnn = nn.LSTM(insize, opt.q_hid_size//2, num_layers=opt.q_layers,
                           bidirectional=True)
        self.pinit = opt.qinit
        self.init_weights()

    def init_weights(self):
        initrange = self.pinit
        rnns = [self.rnn]
        for rnn in rnns:
            for thing in rnn.parameters():
                thing.data.uniform_(-initrange, initrange)

    def forward(self, x):
        """
        x - T x bsz x dim
        returns T x bsz x dim
        """
        return self.rnn(x)[0]


class TrEnc(nn.Module):
    def __init__(self, opt):
        super(TrEnc, self).__init__()
        keydim, valdim = 64, 64
        self.trenc = TransformerEncoder(opt.q_layers, opt.q_heads, opt.q_hid_size,
                                        keydim, valdim, opt.dropout, opt.dropout, 9999999)

    def forward(self, x):
        """
        x - T x bsz x dim
        returns T x bsz x dim
        """
        return self.trenc(x.transpose(0, 1)).transpose(0, 1)


def make_seq_model(opt):
    if "rnn" in opt.infarch:
        enc = RNNEnc(opt)
    elif "transformer" in opt.infarch:
        enc = TrEnc(opt)
    elif "cnn" in opt.infarch:
        enc = CNNEnc(opt)
    return enc

class SeqInfNet(nn.Module):
    """
    puts all the nodes in a sequence
    """
    def __init__(self, nvis, opt):
        super(SeqInfNet, self).__init__()
        self.nvis, self.nhid = nvis, opt.nhid
        self.drop = nn.Dropout(opt.dropout)
        # make a symbolic representation of the nodes. features are idx and vis or hid
        V, H = self.nvis, self.nhid
        nodes = torch.LongTensor(V+H, 2)
        nodes[:, 0].copy_(torch.arange(V+H))
        nodes[:V, 1].fill_(V+H)
        nodes[V:, 1].fill_(V+H+1)
        self.register_buffer("nodes", nodes)

        qemb_sz = opt.qemb_sz
        q_hid_size = opt.q_hid_size
        self.lut = nn.Embedding(V+H+2, qemb_sz)
        self.model = make_seq_model(opt)
        self.decoder = nn.Sequential(ResidualLayer(q_hid_size*2, q_hid_size*2),
                                     ResidualLayer(q_hid_size*2, q_hid_size*2),
                                     self.drop,
                                     nn.Linear(q_hid_size*2, 4))

    def q(self):
        """
        returns V*H x K^2 logits
        """
        V, H = self.nvis, self.nhid
        node_embs = self.lut(self.nodes).view(V+H, 1, -1) # V+H x 1 x 2*qemb_sz

        states = self.model(node_embs) # V+H x 1 x 2*qemb_sz

        vstates, hstates = states[:V], states[V:]

        # make edge reps and get logits: V*H x 4
        logits = self.decoder(
            torch.cat([vstates.expand(V, H, -1).contiguous().view(V*H, -1),
                       hstates.view(1, H, -1).expand(V, H, -1).contiguous().view(V*H, -1)], 1))
        return logits


class SeqJustVisInfNet(nn.Module):
    """
    puts all the nodes in a sequence
    """
    def __init__(self, nvis, opt):
        super(SeqJustVisInfNet, self).__init__()
        self.nvis, self.nhid = nvis, opt.nhid
        self.drop = nn.Dropout(opt.dropout)
        # make a symbolic representation of the nodes. features are idx and vis or hid
        V, H = self.nvis, self.nhid

        qemb_sz = opt.qemb_sz
        q_hid_size = opt.q_hid_size
        self.lut = nn.Embedding(V, qemb_sz)
        self.model = make_seq_model(opt)
        self.decoder = nn.Sequential(ResidualLayer(q_hid_size, q_hid_size),
                                     self.drop,
                                     nn.Linear(q_hid_size, H*4))

    def q(self):
        """
        returns V*H x K^2 logits
        """
        V, H = self.nvis, self.nhid
        node_embs = self.lut.weight[:V].view(V, 1, -1) # V x 1 x qemb_sz

        states = self.model(node_embs) # V x 1 x qemb_sz

        # make edge reps and get logits: V*H x 4
        logits = self.decoder(states.view(V, -1)).view(-1, 4)
        return logits




class TwodJustVisInfNet(nn.Module):
    """
    puts all the nodes in a sequence
    """
    def __init__(self, nvis, opt):
        super(TwodJustVisInfNet, self).__init__()
        self.nvis, self.nhid = nvis, opt.nhid
        self.sqv = int(self.nvis**0.5)
        self.drop = nn.Dropout(opt.dropout)
        # make a symbolic representation of the nodes. features are idx and vis or hid
        V, H = self.nvis, self.nhid

        qemb_sz = opt.qemb_sz
        q_hid_size = opt.q_hid_size
        self.lut = nn.Embedding(V, qemb_sz)
        pad = (opt.kW-1)//2
        bias = False
        mods = [nn.Sequential(nn.Conv2d(qemb_sz, q_hid_size, opt.kW, padding=pad, bias=bias),
                              nn.ReLU(),
                              nn.BatchNorm2d(q_hid_size),
                              nn.MaxPool2d(kernel_size=opt.kW, stride=1, padding=pad))]
        for _ in range(opt.q_layers):
            mods.append(ResBlock(1, opt, dim=2, bias=bias))

        self.model = nn.Sequential(*mods)

        self.decoder = nn.Sequential(ResidualLayer(q_hid_size, q_hid_size),
                                     self.drop,
                                     nn.Linear(q_hid_size, H*4))
        self.pinit = opt.qinit
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -self.pinit, self.pinit)
                m.bias.data.zero_()

    def q(self):
        """
        returns V*H x K^2 logits
        """
        V, H = self.nvis, self.nhid
        # V x qemb_sz -> qemb_sz x V -> qemb_sz x rt V x rtv
        node_embs = self.lut.weight[:V].t().contiguous().view(-1, self.sqv, self.sqv)

        states = self.model(node_embs.unsqueeze(0)) # 1 x hidsz x rtV x rtV

        # make edge reps and get logits: V*H x 4
        logits = self.decoder(states.view(-1, V).t()).view(-1, 4)
        return logits


class DblSeqInfNet(nn.Module):
    def __init__(self, nvis, opt):
        super(DblSeqInfNet, self).__init__()
        self.nvis, self.nhid = nvis, opt.nhid
        self.drop = nn.Dropout(opt.dropout)
        V, H = self.nvis, self.nhid

        qemb_sz = opt.qemb_sz
        q_hid_size = opt.q_hid_size
        self.lut = nn.Embedding(V+H, qemb_sz)
        self.vmodel = make_seq_model(opt)
        self.hmodel = make_seq_model(opt)
        self.decoder = nn.Sequential(ResidualLayer(q_hid_size*2, q_hid_size*2),
                                     ResidualLayer(q_hid_size*2, q_hid_size*2),
                                     self.drop,
                                     nn.Linear(q_hid_size*2, 4))

    def q(self):
        """
        returns V*H x K^2 logits
        """
        V, H = self.nvis, self.nhid
        vembs = self.lut.weight[:V].unsqueeze(1) # V x 1 x qembsz
        hembs = self.lut.weight[V:].unsqueeze(1) # H x 1 x qembsz

        vstates = self.vmodel(vembs) # V x 1 x q_hid_size
        hstates = self.hmodel(hembs) # H x 1 x q_hid_size

        # make edge reps and get logits: V*H x 4
        logits = self.decoder(
            torch.cat([vstates.expand(V, H, -1).contiguous().view(V*H, -1),
                       hstates.view(1, H, -1).expand(V, H, -1).contiguous().view(V*H, -1)], 1))
        return logits
