import argparse
import torch
import numpy as np

from pen_uhmm import HybEdgeModel, RNodeInfNet, TNodeInfNet, inner_lossz, get_taus_and_pens
from utils import get_hmm_stuff, clip_opt_params, batch_kl
from lbp_util import dolbp

parser = argparse.ArgumentParser(description='')
parser.add_argument('-data', type=str, default="/scratch/data/ptb/",
                    help='location of the data corpus')
parser.add_argument('-thresh', type=int, default=0, help='')
parser.add_argument('-max_len', type=int, default=50, help='')

parser.add_argument('-not_residual', action='store_true', help='')
parser.add_argument('-lemb_size', type=int, default=100, help='size of latent label embeddings')
parser.add_argument('-vemb_size', type=int, default=100, help='size of index embeddings')
parser.add_argument('-not_inf_residual', action='store_true', help='')

parser.add_argument('-q_hid_size', type=int, default=100, help='')
parser.add_argument('-q_layers', type=int, default=2, help='')
parser.add_argument('-q_heads', type=int, default=3, help='')
parser.add_argument('-qemb_size', type=int, default=100, help='size of inf index embeddings')
parser.add_argument('-infarch', type=str, default='rnnnode',
                    choices=["rnnnode", "transnode"], help='')

parser.add_argument('-just_diff', action='store_true', help='potentials just look at abs(diff)')
parser.add_argument('-use_length', action='store_true', help='use length as potential feature')
parser.add_argument('-with_idx_ind', action='store_true', help='feature for edges ending at max(1, M-1)')
parser.add_argument('-t_hid_size', type=int, default=100, help='transition hid size')

parser.add_argument('-K', type=int, default=30, help='')
parser.add_argument('-markov_order', type=int, default=1, help='')

parser.add_argument('-optalg', type=str, default='sgd', choices=["sgd", "adagrad", "adam"], help='')
parser.add_argument('-pen_mult', type=float, default=1, help='')
parser.add_argument('-penfunc', type=str, default="l2",
                    choices=["l2", "l1", "js", "kl1", "kl2", "pwl2"], help='')
parser.add_argument('-lbp_iter', type=int, default=10, help='')
parser.add_argument('-lbp_tol', type=float, default=0.001, help='')
parser.add_argument('-randomize_lbp', action='store_true', help='')

parser.add_argument('-init', type=float, default=0.1, help='param init')
parser.add_argument('-qinit', type=float, default=0.1, help='infnet param init')
parser.add_argument('-lr', type=float, default=1, help='initial learning rate')
parser.add_argument('-ilr', type=float, default=1, help='initial learning rate')
parser.add_argument('-inf_iter', type=int, default=1, help='')
parser.add_argument('-decay', type=float, default=0.5, help='initial learning rate')
parser.add_argument('-clip', type=float, default=5, help='gradient clipping')
parser.add_argument('-epochs', type=int, default=40, help='upper epoch limit')
parser.add_argument('-bsz', type=int, default=8, help='batch size')
parser.add_argument('-dropout', type=float, default=0.2, help='dropout')
parser.add_argument('-seed', type=int, default=1111, help='random seed')
parser.add_argument('-cuda', action='store_true', help='use CUDA')

def corr(t1, t2):
    return np.corrcoef(t1.data.cpu().numpy(), t2.data.cpu().numpy())[0][1]

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ntypes, max_verts, max_seqlen = 10002, 3, 30
    model = HybEdgeModel(ntypes, max_verts, args).to(device)
    if args.infarch == "rnnnode":
        infnet = RNodeInfNet(ntypes, max_seqlen, args).to(device)
    else:
        infnet = TNodeInfNet(ntypes, max_seqlen, args).to(device)

    T, M, K = 10, 3, 30
    edges, nodeidxs, ne = get_hmm_stuff(T, M, K)
    edges, ne = edges.to(device), ne.view(1, -1).to(device)
    nodeidxs = nodeidxs.to(device)
    npenterms = (nodeidxs != 2*edges.size(0)).sum().float()
    nedges = edges.size(0)
    EPS = 1e-38
    neginf = torch.Tensor(1, 1, 1).fill_(-1e18).to(device)

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

    if args.optalg == "sgd":
        popt = torch.optim.SGD(
            [#{"params": model.parameters(), "lr": args.lr},
             {"params": infnet.parameters(), "lr": args.ilr}])
    else:
        popt = torch.optim.Adam([#{"params": model.parameters(), "lr": args.lr},
                                 {"params": infnet.parameters(), "lr": args.ilr}])

    with torch.no_grad():
        #un_lpots = model.get_obs_lps(batch) # bsz x T x K log unary potentials
        ed_lpots = model.get_edge_scores(edges, T) # nedges x K*K log potentials

        exed_lpots = ed_lpots.view(nedges, 1, K, K)
        # get approximate unclamped marginals
        nodebs, facbs, ii, _, _ = dolbp(exed_lpots, edges, niter=args.lbp_iter, renorm=True,
                                        randomize=args.randomize_lbp, tol=args.lbp_tol)

        tau_u = torch.stack([nodebs[t] for t in range(T)]).transpose(0, 1) # 1 x T x K
        tau_e = torch.stack([facbs[e] for e in range(nedges)]).transpose(0, 1) # 1 x nedges x K x K
        tau_u, tau_e = (tau_u.exp() + EPS), (tau_e.exp() + EPS)

    for i in range(args.inf_iter):
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

        popt.zero_grad()
        pred_rho = infnet.q(edges, T) # nedges x K^2 logits
        in_loss, ipen_loss = inner_lossz(pred_rho.view(1, -1), ed_lpots.view(1, -1), nodeidxs,
                                         K, ne, neginf, penfunc)
        in_loss = in_loss + args.pen_mult/npenterms * ipen_loss
        print("in_loss", in_loss.item())
        in_loss.backward()
        clip_opt_params(popt, args.clip)
        popt.step()
