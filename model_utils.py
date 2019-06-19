import torch.nn as nn
import torch.nn.functional as F

class ResidualLayer(nn.Module):
    def __init__(self, in_dim=100, out_dim=100):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x

# 1d res block w/ optional dilation
class ResBlock(nn.Module):
    def __init__(self, dil, opt, dim=1, bias=True):
        super(ResBlock, self).__init__()
        pad = (opt.kW-1)//2
        if dim == 1:
            self.conv1 = nn.Conv1d(opt.q_hid_size, opt.q_hid_size, opt.kW,
                                   padding=pad*dil, dilation=dil, bias=bias)
            self.bn1 = nn.BatchNorm1d(opt.q_hid_size)
            # i guess this one isn't dilated???
            self.conv2 = nn.Conv1d(opt.q_hid_size, opt.q_hid_size, opt.kW,
                                   padding=pad, bias=bias)
            self.bn2 = nn.BatchNorm1d(opt.q_hid_size)
        else:
            # i think often these don't have a bias?
            self.conv1 = nn.Conv2d(opt.q_hid_size, opt.q_hid_size, opt.kW,
                                   padding=pad*dil, dilation=dil, bias=bias)
            self.bn1 = nn.BatchNorm2d(opt.q_hid_size)
            # i guess this one isn't dilated???
            self.conv2 = nn.Conv2d(opt.q_hid_size, opt.q_hid_size, opt.kW,
                                   padding=pad, bias=bias)
            self.bn2 = nn.BatchNorm2d(opt.q_hid_size)

    def forward(self, x):
        return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
