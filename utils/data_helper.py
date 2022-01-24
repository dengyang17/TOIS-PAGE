import pickle as pkl
import linecache
from random import Random

import numpy as np
import torch
import torch.utils.data as torch_data

import utils

num_samples = 1


class BiTestDataset(torch_data.Dataset):

    def __init__(self, infos, indices=None):

        self.src_id = infos['src_id']
        self.src_str = infos['src_str']
        self.tgt_id = infos['tgt_id']
        self.tgt_str = infos['tgt_str']
        self.length = infos['length']
        self.infos = infos
        if indices is None:
            self.indices = list(range(self.length))
        else:
            self.indices = indices

    def __getitem__(self, index):
        index = self.indices[index]
        src = self.src_id[index]
        original_src = self.src_str[index]
        tgt = self.tgt_id[index]
        original_tgt = self.tgt_str[index]

        return src, tgt, original_src, original_tgt

    def __len__(self):
        return len(self.indices)


class QADataset(torch_data.Dataset):

    def __init__(self, infos, indices=None, char=False):

        self.srcF = infos['srcF']
        self.tgtF = infos['tgtF']
        self.factF = infos['factF']
        self.original_srcF = infos['original_srcF']
        self.original_tgtF = infos['original_tgtF']
        self.original_factF = infos['original_factF']
        self.length = infos['length']
        self.infos = infos
        self.char = char
        if indices is None:
            self.indices = list(range(self.length))
        else:
            self.indices = indices

    def __getitem__(self, index):
        index = self.indices[index]
        src = list(map(int, linecache.getline(
            self.srcF, index+1).strip().split()))
        tgt = list(map(int, linecache.getline(
            self.tgtF, index+1).strip().split()))
        original_src = linecache.getline(
            self.original_srcF, index+1).strip().split()
        original_tgt = linecache.getline(self.original_tgtF, index+1).strip().split() if not self.char else \
            list(linecache.getline(self.original_tgtF, index + 1).strip())
        
        fact = eval(linecache.getline(self.factF, index+1))
        original_fact = eval(linecache.getline(self.original_factF, index+1))

        return src, tgt, original_src, original_tgt, fact, original_fact

    def __len__(self):
        return len(self.indices)


class PersonaQADataset(QADataset):

    def __init__(self, persona_id_path, persona_str_path, vocab, **kwargs):
        QADataset.__init__(self, **kwargs)
        self.persona_id_path = persona_id_path
        self.persona_str_path = persona_str_path
        self.vocab = vocab

    def __getitem__(self, index):
        src, tgt, original_src, original_tgt, fact, original_fact = QADataset.__getitem__(self, index)
        persona = eval(linecache.getline(self.persona_id_path, index+1))
        original_persona = eval(linecache.getline(self.persona_str_path, index+1))
        persBow = self.vocab.convertToBow([y for x in persona for y in x])
        return src, tgt, original_src, original_tgt, fact, persona, original_fact, original_persona, persBow


def splitDataset(data_set, sizes):
    length = len(data_set)
    indices = list(range(length))
    rng = Random()
    rng.seed(1234)
    rng.shuffle(indices)

    data_sets = []
    part_len = int(length / sizes)
    for i in range(sizes-1):
        data_sets.append(QADataset(data_set.infos, indices[0:part_len]))
        indices = indices[part_len:]
    data_sets.append(QADataset(data_set.infos, indices))
    return data_sets

def padding(data):
    src, tgt, original_src, original_tgt, fact, original_fact = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s[:end])

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s)[:end]
    
    fact_len = [[len(ff) for ff in f] for f in fact]
    fact_num = [len(f) for f in fact]
    fact_pad = torch.zeros(len(fact), max(fact_num), max([max(l) for l in fact_len])).long()
    for i, f in enumerate(fact):
        for j, s in enumerate(f):
            end = fact_len[i][j]
            fact_pad[i, j, :end] = torch.LongTensor(s)[:end]

    return src_pad, tgt_pad, \
           original_src, original_tgt, fact_pad, None, original_fact, None, None

def persona_padding(data):
    src, tgt, original_src, original_tgt, fact, pers, original_fact, original_persona, pers_bow = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s[:end])

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s)[:end]

    fact_len = [[len(ff) for ff in f] for f in fact]
    fact_num = [len(f) for f in fact]
    fact_pad = torch.zeros(len(fact), max(fact_num), max([max(l) for l in fact_len])).long()
    for i, f in enumerate(fact):
        for j, s in enumerate(f):
            end = fact_len[i][j]
            fact_pad[i, j, :end] = torch.LongTensor(s)[:end]
    
    pers_len = [[len(pp) for pp in p] for p in pers]
    pers_num = [len(p) for p in pers]
    if max(pers_num) == 0:
        pers_pad = torch.zeros(len(pers), 1, 1).long()
    else:
        pers_pad = torch.zeros(len(pers), max(pers_num), max([max(l) for l in pers_len if len(l) > 0])).long()
        for i, p in enumerate(pers):
            for j, s in enumerate(p):
                end = pers_len[i][j]
                pers_pad[i, j, :end] = torch.LongTensor(s)[:end]
    pers_bow = torch.FloatTensor(pers_bow)

    return src_pad, tgt_pad, \
           original_src, original_tgt, fact_pad, pers_pad, original_fact, original_persona, pers_bow


def split_padding(data):
    src, tgt, original_src, original_tgt = zip(*data)

    split_samples = []
    num_per_sample = int(len(src) / utils.num_samples)

    for i in range(utils.num_samples):
        split_src = src[i*num_per_sample:(i+1)*num_per_sample]
        split_tgt = tgt[i*num_per_sample:(i+1)*num_per_sample]
        split_original_src = original_src[i *
                                          num_per_sample:(i + 1) * num_per_sample]
        split_original_tgt = original_tgt[i *
                                          num_per_sample:(i + 1) * num_per_sample]

        src_len = [len(s) for s in split_src]
        src_pad = torch.zeros(len(split_src), max(src_len)).long()
        for i, s in enumerate(split_src):
            end = src_len[i]
            src_pad[i, :end] = torch.LongTensor(s)[:end]

        tgt_len = [len(s) for s in split_tgt]
        tgt_pad = torch.zeros(len(split_tgt), max(tgt_len)).long()
        for i, s in enumerate(split_tgt):
            end = tgt_len[i]
            tgt_pad[i, :end] = torch.LongTensor(s)[:end]

        split_samples.append([src_pad, tgt_pad,
                              torch.LongTensor(
                                  src_len), torch.LongTensor(tgt_len),
                              split_original_src, split_original_tgt])

    return split_samples
