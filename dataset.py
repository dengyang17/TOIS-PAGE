import os
import pickle

import torch

import utils


def load_data(config):
    """
    load data.
    update "data" due to the saved path in the pickle file
    :return: a dict with data and vocabulary
    """
    print("loading data...\n")
    data = pickle.load(open(config.data + "data.pkl", "rb"))
    # retrieve data, due to the problem of path.
    data["train"]["length"] = int(data["train"]["length"] * config.scale)
    data["train"]["srcF"] = os.path.join(config.data, "train.src.id")
    data["train"]["original_srcF"] = os.path.join(config.data, "train.src.str")
    data["train"]["tgtF"] = os.path.join(config.data, "train.tgt.id")
    data["train"]["original_tgtF"] = os.path.join(config.data, "train.tgt.str")
    data["train"]["factF"] = os.path.join(config.data, "train.fact.id")
    data["train"]["original_factF"] = os.path.join(config.data, "train.fact.str")
    data["test"]["srcF"] = os.path.join(config.data, "test.src.id")
    data["test"]["original_srcF"] = os.path.join(config.data, "test.src.str")
    data["test"]["tgtF"] = os.path.join(config.data, "test.tgt.id")
    data["test"]["original_tgtF"] = os.path.join(config.data, "test.tgt.str")
    data["test"]["factF"] = os.path.join(config.data, "test.fact.id")
    data["test"]["original_factF"] = os.path.join(config.data, "test.fact.str")

    src_vocab = data["dict"]["src"]
    tgt_vocab = data["dict"]["tgt"]
    config.src_vocab_size = src_vocab.size()
    config.tgt_vocab_size = tgt_vocab.size()

    if config.persona:
        train_set = utils.PersonaQADataset(
            os.path.join(config.data, 'train.pers.id'), os.path.join(config.data, 'train.pers.str'),
            infos=data['train'], vocab=src_vocab, char=config.char)
        valid_set = utils.PersonaQADataset(
            os.path.join(config.data, 'test.pers.id'), os.path.join(config.data, 'test.pers.str'),
            infos=data['test'], vocab=src_vocab, char=config.char)
    else:
        train_set = utils.QADataset(data["train"], char=config.char)
        valid_set = utils.QADataset(data["test"], char=config.char)


    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.persona_padding if config.persona else utils.padding,
    )
    if hasattr(config, "valid_batch_size"):
        valid_batch_size = config.valid_batch_size
    else:
        valid_batch_size = config.batch_size
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_set,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.persona_padding if config.persona else utils.padding,
    )
    return {
        "train_set": train_set,
        "valid_set": valid_set,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
    }


def extend_vocab(vocab, original_src, original_tgt, original_fact, original_pers, src, tgt, fact, pers):
    oovs = None
    src_extend_ids = torch.zeros(src.size()).long()
    for i, s in enumerate(original_src):
        sid, oovs = vocab.convertToIdxandOOVs(s, utils.UNK_WORD, oovs=oovs)
        end = sid.size(0)
        src_extend_ids[i, :end] = sid
    
    tgt_extend_ids = torch.zeros(tgt.size()).long()
    for i, s in enumerate(original_tgt):
        sid, oovs = vocab.convertToIdxandOOVs(s, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD, oovs=oovs)
        end = sid.size(0)
        tgt_extend_ids[i, :end] = sid
    
    fact_extend_ids = torch.zeros(fact.size()).long()
    for i, f in enumerate(original_fact):
        for j, s in enumerate(f):
            sid, oovs = vocab.convertToIdxandOOVs(s.split(' '), utils.UNK_WORD, oovs=oovs)
            end = sid.size(0)
            fact_extend_ids[i, j, :end] = sid
    
    pers_extend_ids = None
    if pers is not None:
        pers_extend_ids = torch.zeros(pers.size()).long()
        for i, p in enumerate(original_pers):
            for j, s in enumerate(p):
                sid, oovs = vocab.convertToIdxandOOVs(s.split(' '), utils.UNK_WORD, oovs=oovs)
                end = sid.size(0)
                pers_extend_ids[i, j, :end] = sid
    
    return src_extend_ids, tgt_extend_ids, fact_extend_ids, pers_extend_ids, oovs