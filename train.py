import codecs
import json
import os
import pickle
import random
import time
from argparse import Namespace
from collections import OrderedDict, defaultdict
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import yaml
from torch.nn.init import xavier_uniform_
from tqdm import tqdm

import lr_scheduler as L
import models
import opts
import utils
from dataset import load_data, extend_vocab
from utils import misc_utils


# python train.py --config=configs/config.yaml --expname=expname
# python train.py --config=configs/config.yaml --expname=expname --mode=eval --restore=experiments/expname/checkpoint.pt --beam-size=10 --metrics="bleu rouge"


# build model
def build_model(checkpoints, config, device, devices_id):
    """
    build model, either Seq2Seq or Tensor2Tensor
    :param checkpoints: load checkpoint if there is pretrained model
    :return: model, optimizer and the print function
    """
    print(config)

    # model
    print("building model...\n")
    model = getattr(models, config.model)(
        config, device,
        src_padding_idx=utils.PAD,
        tgt_padding_idx=utils.PAD,
        label_smoothing=config.label_smoothing,
    )
    if len(devices_id) > 1:
        print(devices_id)
        model = torch.nn.DataParallel(model, device_ids=devices_id)
    model.to(device)
    if config.param_init != 0.0:
        for p in model.parameters():
            p.data.uniform_(-config.param_init, config.param_init)
    if config.param_init_glorot:
        for p in model.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    if checkpoints is not None:
        model.load_state_dict(checkpoints["model"])
    if config.pretrain:
        print("loading checkpoint from %s" % config.pretrain)
        pre_ckpt = torch.load(
            config.pretrain, map_location=lambda storage, loc: storage
        )["model"]
        model.load_state_dict(pre_ckpt)

    optim = models.Optim(
        config.optim,
        config.learning_rate,
        config.max_grad_norm,
        lr_decay=config.learning_rate_decay,
        start_decay_steps=config.start_decay_steps,
        beta1=config.beta1,
        beta2=config.beta2,
        decay_method=config.decay_method,
        warmup_steps=config.warmup_steps,
        model_size=config.hidden_size,
    )
    print(optim)
    optim.set_parameters(model.parameters())
    if checkpoints is not None:
        optim.optimizer.load_state_dict(checkpoints["optim"])

    param_count = sum([param.view(-1).size()[0] for param in model.parameters()])
    print(repr(model) + "\n\n")
    print("total number of parameters: %d\n\n" % param_count)

    return model, optim


def train_model(model, data, optim, epoch, params, config, device, writer):
    model.train()
    train_loader = data["train_loader"]
    log_vars = defaultdict(float)
    src_vocab = data["src_vocab"]
    logging.basicConfig(level=logging.DEBUG, filename=params["log_path"] + 'log.txt', filemode='a')

    for src, tgt, original_src, original_tgt, fact, pers, original_fact, original_pers, pers_bow in tqdm(train_loader):

        src_extend_ids, tgt_extend_ids, fact_extend_ids, pers_extend_ids, oovs = extend_vocab(src_vocab,
                                 original_src, original_tgt, original_fact, original_pers, src, tgt, fact, pers)
        bow_ids = src_vocab.get_bow_ids()
        bow_ids = torch.LongTensor(bow_ids).expand(src.size(0), len(bow_ids)).to(device)

        # put the tensors on cuda devices
        #max_ext_len = torch.LongTensor([[len(oovs)]]).expand(src.size(0), 1).to(device)
        max_ext_len = len(oovs)

        src, tgt, fact = src.to(device), tgt.to(device), fact.to(device)
        src_extend_ids, tgt_extend_ids, fact_extend_ids =  src_extend_ids.to(device), tgt_extend_ids.to(device), fact_extend_ids.to(device)
        
        if config.persona:
            pers, pers_extend_ids, pers_bow = pers.to(device), pers_extend_ids.to(device), pers_bow.to(device)

        model.zero_grad()

        dec = tgt[:, :-1]  # [batch, len]
        targets = tgt[:, 1:]  # [batch, len]
        tgt_extend_ids = tgt_extend_ids[:, 1:]

        try:
            return_dict, outputs = model(
                src, dec, targets, fact, pers, pers_bow, src_extend_ids, tgt_extend_ids, fact_extend_ids, pers_extend_ids, bow_ids, max_ext_len
            )
            # outputs: [batch, len, size]
            pred = outputs.max(2)[1]
            
            if config.pointer:
                num_correct = (
                    pred.eq(tgt_extend_ids).masked_select(tgt_extend_ids.ne(utils.PAD)).sum().item()
                )
                num_total = tgt_extend_ids.ne(utils.PAD).sum().item()
            else:
                num_correct = (
                    pred.eq(targets).masked_select(targets.ne(utils.PAD)).sum().item()
                )
                num_total = targets.ne(utils.PAD).sum().item()

            return_dict["mle_loss"] = torch.sum(return_dict["mle_loss"].mean()) / num_total
            if config.vae:
                return_dict["vae_loss"] = torch.sum(return_dict["vae_loss"].mean()) / num_total
                return_dict["total_loss"] = return_dict["mle_loss"] + return_dict["vae_loss"]

            else:
                return_dict["total_loss"] = return_dict["mle_loss"]
            return_dict["total_loss"].backward()
            optim.step()

            for key in return_dict:
                log_vars[key] += return_dict[key].item()
            params["report_total_loss"] += return_dict["total_loss"].item()
            params["report_correct"] += num_correct
            params["report_total"] += num_total

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory")
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
            else:
                raise e

        # utils.progress_bar(params['updates'], config.eval_interval)
        params["updates"] += 1
        #print("loss after %d steps: %f\r" % (params["updates"], return_dict["total_loss"].item()))

        if params["updates"] % config.report_interval == 0:
            logging.info(
                "epoch: %3d, loss: %6.3f, time: %6.3f, updates: %8d, accuracy: %2.2f\n"
                % (
                    epoch,
                    params["report_total_loss"] / config.report_interval,
                    time.time() - params["report_time"],
                    params["updates"],
                    params["report_correct"] * 100.0 / params["report_total"],
                )
            )

            for key in return_dict:
                writer.add_scalar(
                    f"train/{key}",
                    log_vars[key] / config.report_interval,
                    params["updates"],
                )
            # writer.add_scalar("train" + "/lr", optim.lr, params['updates'])
            writer.add_scalar(
                "train" + "/accuracy",
                params["report_correct"] / params["report_total"],
                params["updates"],
            )

            log_vars = defaultdict(float)
            params["report_total_loss"], params["report_time"] = 0, time.time()
            params["report_correct"], params["report_total"] = 0, 0

    if epoch % config.eval_interval == 0:
        logging.info("evaluating after %d updates...\r" % params["updates"])
        score = eval_model(model, data, params, config, device, writer)
        for metric in config.metrics:
            params[metric].append(score[metric])
            if score[metric] >= max(params[metric]):
                with codecs.open(
                    params["log_path"] + "best_" + metric + "_prediction.txt",
                    "w",
                    "utf-8",
                ) as f:
                    f.write(
                        codecs.open(
                            params["log_path"] + "candidate.txt", "r", "utf-8"
                        ).read()
                    )
                save_model(
                    params["log_path"] + "best_" + metric + "_checkpoint.pt",
                    model,
                    optim,
                    params["updates"],
                    config,
                )
                logging.info("best model saved...")
            writer.add_scalar(
                "valid" + "/" + metric, score[metric], params["updates"]
            )
        model.train()

    if epoch % config.save_interval == 0:
        if config.save_individual:
            save_model(
                        params["log_path"] + str(params["updates"]) + "checkpoint.pt",
                        model,
                        optim,
                        params["updates"],
                        config,
            )
        save_model(
                    params["log_path"] + "checkpoint.pt",
                    model,
                    optim,
                    params["updates"],
                    config,
        )

    if config.epoch_decay:
        optim.updateLearningRate(epoch)


def eval_model(model, data, params, config, device, writer):
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    # count, total_count = 0, len(data['valid_set'])
    valid_loader = data["valid_loader"]
    tgt_vocab = data["tgt_vocab"]
    logging.basicConfig(level=logging.DEBUG, filename=params["log_path"] + 'log.txt', filemode='a')

    for src, tgt, original_src, original_tgt, fact, pers, original_fact, original_pers, pers_bow in tqdm(valid_loader):
        src_extend_ids, tgt_extend_ids, fact_extend_ids, pers_extend_ids, oovs = extend_vocab(tgt_vocab,
                                 original_src, original_tgt, original_fact, original_pers, src, tgt, fact, pers)

        bow_ids = tgt_vocab.get_bow_ids()
        bow_ids = torch.LongTensor(bow_ids).expand(src.size(0), len(bow_ids)).to(device)

        # put the tensors on cuda devices
        #max_ext_len = torch.LongTensor([[len(oovs)]]).expand(src.size(0), 1).to(device)
        max_ext_len = len(oovs)

        src, fact = src.to(device), fact.to(device)
        src_extend_ids, fact_extend_ids =  src_extend_ids.to(device), fact_extend_ids.to(device)

        if config.persona:
            pers, pers_extend_ids, pers_bow = pers.to(device), pers_extend_ids.to(device), pers_bow.to(device)

        with torch.no_grad():
            if config.beam_size > 1:
                samples, alignment = model.module.beam_sample(
                    src, fact, pers, pers_bow, src_extend_ids, fact_extend_ids, pers_extend_ids, bow_ids, max_ext_len, beam_size=config.beam_size, eval_=False
                )
            else:
                samples, alignment = model.module.sample(src, fact, pers, pers_bow, src_extend_ids, fact_extend_ids, pers_extend_ids, bow_ids, max_ext_len)
        if config.pointer:
            candidate += [tgt_vocab.convertToLabels(s, utils.EOS, oovs) for s in samples]
        else:
            candidate += [tgt_vocab.convertToLabels(s, utils.EOS) for s in samples]
        source += original_src
        reference += original_tgt
        if alignment is not None:
            alignments += [align for align in alignment]

    if config.unk and config.copy:
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == utils.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
            if len(cand) == 0:
                print("Error!")
        candidate = cands

    with codecs.open(
        os.path.join(params["log_path"], "candidate.txt"), "w+", "utf-8"
    ) as f:
        for i in range(len(candidate)):
            f.write(f"{' '.join(candidate[i])}\n")
    if config.label_dict_file != "":
        results = utils.eval_metrics(
            reference, candidate, label_dict, params["log_path"]
        )
    score = {}
    result_line = ""
    for metric in config.metrics:
        if config.label_dict_file != "":
            score[metric] = results[metric]
            result_line += metric + ": %s " % str(score[metric])
        else:
            score[metric] = getattr(utils, metric)(
                reference, candidate, params["log_path"], logging.info, config
            )
            score[metric] = getattr(utils, metric)(
                reference, candidate, params["log_path"], print, config
            )
    
    if config.vae:
        model.module.npm.print_topic_words(tgt_vocab, os.path.join(params["log_path"] + 'topwords.txt'))

    if config.label_dict_file != "":
        result_line += "\n"
        print(result_line)

    return score


# save model
def save_model(path, model, optim, updates, config):
    model_state_dict = model.state_dict()
    optim_state_dict = optim.optimizer.state_dict()
    checkpoints = {
        "model": model_state_dict,
        "config": config,
        "updates": updates,
        "optim": optim_state_dict,
    }
    torch.save(checkpoints, path)


if __name__ == "__main__":
    # Combine command-line arguments and yaml file arguments
    opt = opts.model_opts()
    config = yaml.load(open(opt.config, "r"))
    config = Namespace(**config, **vars(opt))
    config.metrics = config.metrics.split(' ')

    writer = misc_utils.set_tensorboard(config)
    device, devices_id = misc_utils.set_cuda(config)
    misc_utils.set_seed(config.seed)

    if config.label_dict_file:
        with open(config.label_dict_file, "r") as f:
            label_dict = json.load(f)

    if config.restore:
        print("loading checkpoint...\n")
        checkpoints = torch.load(
            config.restore, map_location=lambda storage, loc: storage
        )
    else:
        checkpoints = None

    data = load_data(config)
    model, optim = build_model(checkpoints, config, device, devices_id)
    if config.schedule:
        scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)

    params = {
        "updates": 0,
        "report_total_loss": 0,
        "report_total": 0,
        "report_correct": 0,
        "report_time": time.time(),
        "log_path": os.path.join(config.logdir, config.expname) + "/",
    }
    for metric in config.metrics:
        params[metric] = []
    if config.restore:
        params["updates"] = checkpoints["updates"]

    if config.mode == "train":
        for i in range(1, config.epoch + 1):
            if config.schedule:
                scheduler.step()
                print("Decaying learning rate to %g" % scheduler.get_lr()[0])
            train_model(model, data, optim, i, params, config, device, writer)
        for metric in config.metrics:
            print("Best %s score: %.3f\n" % (metric, max(params[metric])))
    else:
        score = eval_model(model, data, params, config, device, writer)
