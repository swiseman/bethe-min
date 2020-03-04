Code for [Amortized Bethe Free Energy Minimization for
Learning MRFs](https://arxiv.org/pdf/1906.06399.pdf).

## Ising model
To compare marginals for a 10x10 Ising model averaged across 5 iterations:
```
python ising_marginals.py --gpu 0 --n 10 --exp_iters 5
```

## RBM
To train the RBM with amortized BFE, run:
```
python rbm.py -cuda -epochs 40 -ilr 0.003 -log_interval 200 -lr 0.001 -optalg adam -pen_mult 1.5 -q_hid_size 150 -q_layers 5 -qemb_sz 200 -seed 72831 -save rbm-model.pt
```

To run AIS:
```
python rbm.py -cuda -train_from rbm-model.pt
```

## Undirected HMM
To train the undirected HMM variant with amortized BFE, run:
```
python pen_uhmm.py -cuda -K 30 -bsz 32 -dropout 0.3 -ilr 0.0003 -infarch rnnnode -init 0.001 -just_diff -lemb_size 64 -log_interval 500 -loss alt3 -lr 0.0001 -markov_order 3 -max_len 30 -not_inf_residual -optalg adam -pen_mult 1 -pendecay 1 -penfunc kl2 -q_hid_size 100 -q_layers 1 -qemb_size 150 -qinit 0.001 -seed 48047 -t_hid_size 100 -vemb_size 64 -wemb_size 100 -epochs 10
```
