"""
this file modified from the word_language_model example
"""
import os
import torch

from collections import Counter, defaultdict

import torch

class Dictionary(object):
    def __init__(self, unk_word="<unk>"):
        self.unk_word = unk_word
        self.idx2word = [unk_word, "<pad>", "<bos>", "<eos>"] # OpenNMT constants

    def add_word(self, word, train=False):
        """
        returns idx of word
        """
        if train and word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word] if word in self.word2idx else self.word2idx[self.unk_word]

    def bulk_add(self, words):
        """
        assumes train=True
        """
        self.idx2word.extend(words)
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def __len__(self):
        return len(self.idx2word)


class SentCorpus(object):
    def __init__(self, path, bsz, vocabsize=0, thresh=0, min_len=1, max_len=10000, vocab=None):
        self.bsz = bsz
        self.dictionary = Dictionary()

        if vocab is None:
            self.get_vocabs(os.path.join(path, "train.txt"), vocabsize=vocabsize, thresh=thresh)
        else:
            self.dictionary.idx2word = vocab
            self.dictionary.word2idx = {wrd: i for i, wrd in enumerate(vocab)}
        print("using vocabulary of size:", len(self.dictionary))

        with open(os.path.join(path, "train.txt")) as f:
            # get sentence, lineno pairs
            toktrsents = []
            for i, line in enumerate(f):
                toks = [self.dictionary.add_word(wrd, train=False) for wrd in line.strip().split()]
                if len(toks) >= min_len and len(toks) <= max_len-1:
                    toks.append(self.dictionary.word2idx["<eos>"])
                    toktrsents.append((toks, i))
            # toktrsents = [([self.dictionary.add_word(wrd, train=False)
            #                 for wrd in line.strip().split()], i)
            #               for i, line in enumerate(f)]
            # # add on eoses
            # [tup[0].append(self.dictionary.word2idx["<eos>"]) for tup in toktrsents]
        self.train, self.train_mb2linenos = self.minibatchify(toktrsents, bsz) # list of minibatches

        with open(os.path.join(path, "valid.txt")) as f:
            # get sentence, lineno pairs
            tokvalsents = []
            for i, line in enumerate(f):
                toks = [self.dictionary.add_word(wrd, train=False) for wrd in line.strip().split()]
                if len(toks) >= min_len and len(toks) <= max_len-1:
                    toks.append(self.dictionary.word2idx["<eos>"])
                    tokvalsents.append((toks, i))
            # tokvalsents = [([self.dictionary.add_word(wrd, train=False)
            #                  for wrd in line.strip().split()], i)
            #                for i, line in enumerate(f)]
            # # add on eoses
            # [tup[0].append(self.dictionary.word2idx["<eos>"]) for tup in tokvalsents]
        self.valid, self.val_mb2linenos = self.minibatchify(tokvalsents, bsz)


    def get_vocabs(self, fi, vocabsize=70000, thresh=0):
        tgt_voc = Counter()
        with open(fi) as f:
            for line in f:
                sent = line.strip().split()
                tgt_voc.update(sent)

        # delete special tokens
        for key in self.dictionary.idx2word[:4]:
            if key in tgt_voc:
                del tgt_voc[key]

        if thresh > 0:
            for k in list(tgt_voc.keys()):
                if tgt_voc[k] <= thresh:
                    del tgt_voc[k]
            vocab = [tup[0] for tup in tgt_voc.items()]
        elif vocabsize > 0:
            vocab = [tup[0] for tup in tgt_voc.most_common(vocabsize)]
        else:
            vocab = tgt_voc.keys()
        self.dictionary.bulk_add(vocab)


    def minibatchify(self, sents, bsz):
        """
        sents is a list of (sent, lineno) tuples
        """
        # first shuffle
        perm = torch.randperm(len(sents))
        sents = [sents[idx.item()] for idx in perm]
        #random.shuffle(sents)

        # sort in ascending order
        sents, sorted_idxs = zip(*sorted(zip(sents, range(len(sents))),
                                         key=lambda x: len(x[0][0])))
        minibatches, mb2linenos = [], []
        curr_batch, curr_linenos = [], []
        curr_len = len(sents[0][0])
        for i in range(len(sents)):
            if len(sents[i][0]) != curr_len or len(curr_batch) == bsz: # we're done
                minibatches.append(torch.LongTensor(curr_batch).t().contiguous())
                mb2linenos.append(curr_linenos)
                curr_batch = [sents[i][0]]
                curr_len = len(sents[i][0])
                curr_linenos = [sents[i][1]]
            else:
                curr_batch.append(sents[i][0])
                curr_linenos.append(sents[i][1])
        # catch last
        if len(curr_batch) > 0:
            minibatches.append(torch.LongTensor(curr_batch).t().contiguous())
            mb2linenos.append(curr_linenos)
        return minibatches, mb2linenos
