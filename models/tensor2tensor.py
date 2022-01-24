import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models
import utils

# from fairseq import bleu
# from utils.reward_provider import CTRRewardProvider


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


class LabelSmoothingLoss(nn.Module):
    """ Label smoothing loss """
    def __init__(self, device, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()
        self.size = tgt_vocab_size
        self.device = device
        self.smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), self.smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        real_size = output.size(1)
        if real_size > self.size:
            real_size -= self.size
        else:
            real_size = 0

        model_prob = self.one_hot.repeat(target.size(0), 1)
        if real_size > 0:
            ext_zeros = torch.full((model_prob.size(0), real_size), self.smoothing_value).to(self.device)
            model_prob = torch.cat((model_prob, ext_zeros), -1)

        #model_prob = self.one_hot.repeat(target.size(0), 1)
        #if extra_zeros is not None:
        #    extra_zeros = extra_zeros.contiguous().view(-1, extra_zeros.size(2)) 
        #    extra_zeros += self.smoothing_value
        #    model_prob = torch.cat((model_prob, extra_zeros), -1)

        #output = F.log_softmax(output, dim=-1)
        #model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class tensor2tensor(nn.Module):
    """ transformer model """
    def __init__(self, config, device, use_attention=True,
                 encoder=None, decoder=None,
                 src_padding_idx=0, tgt_padding_idx=0,
                 label_smoothing=0, tgt_vocab=None):
        """
        Initialization of variables and functions
        :param config: configuration
        :param use_attention: use attention or not, consistent with seq2seq
        :param encoder: encoder
        :param decoder: decoder
        :param src_padding_idx: source padding index
        :param tgt_padding_idx: target padding index
        :param label_smoothing: ratio for label smoothing
        :param tgt_vocab: target vocabulary
        """
        super(tensor2tensor, self).__init__()

        self.config = config
        self.device = device

        # pretrained encoder or not
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = models.TransformerEncoder(
                config, padding_idx=src_padding_idx)
            #self.fact_encoder = models.TransformerEncoder(
            #    config, padding_idx=src_padding_idx)
            self.fact_encoder = self.encoder
            if self.config.persona:
                #self.persona_encoder = models.TransformerEncoder(
                #    config, padding_idx=src_padding_idx)
                self.persona_encoder = self.encoder

        self.condition_context_attn = models.BiAttention(config.hidden_size, config.dropout)
        self.bi_attn_transform = nn.Linear(config.hidden_size * 4, config.hidden_size)

        if self.config.persona:
            if self.config.experience:
                self.persona_condition_context_attn = models.BiAttention(config.hidden_size, config.dropout)
                self.persona_bi_attn_transform = nn.Linear(config.hidden_size * 4, config.hidden_size)
            if self.config.vae:
                # persona modeling
                self.npm = models.NPM(config)

        tgt_embedding = self.encoder.embedding if config.shared_vocab else None
        # pretrained decoder or not
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = models.TransformerDecoder(
                config, tgt_embedding=tgt_embedding, padding_idx=tgt_padding_idx)
        # log softmax should specify dimension explicitly
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.use_cuda = config.use_cuda
        self.config = config
        self.label_smoothing = label_smoothing
        if self.label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(device,
                label_smoothing, config.tgt_vocab_size,
                ignore_index=tgt_padding_idx)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=utils.PAD)
        if config.use_cuda:
            self.criterion.to(self.device)
        self.compute_score = nn.Linear(
            config.hidden_size, config.tgt_vocab_size)

        self.padding_idx = tgt_padding_idx

    def compute_loss(self, scores, targets):
        """
        loss computation
        :param scores: predicted scores
        :param targets: targets
        :return: loss
        """
        scores = scores.contiguous().view(-1, scores.size(2))   #[batch*len, vocab]
        loss = self.criterion(scores, targets.contiguous().view(-1))
        return loss
    
    # Reconstruction + KL divergence losses summed over all elements and batch
    def VAE_loss(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
    
    def pointer_network(self, outputs, attn, fact_attn, persona_attn, topic_attn, pointers, src_extend_ids, fact_extend_ids, pers_extend_ids, bow_ids, max_ext_len):
        bsz, output_len, _ = outputs.size()
        vocab_dist = F.softmax(outputs, dim=-1)
        if max_ext_len > 0:
            extra_zeros = Variable(torch.zeros(bsz, output_len, max_ext_len)).to(self.device)
            vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1) 
        vocab_dist = vocab_dist * pointers[:,:,0].unsqueeze(-1)
        src_len = src_extend_ids.size(1)
        fact_len = fact_extend_ids.size(1)
        vocab_dist = vocab_dist.scatter_add(2, src_extend_ids.unsqueeze(1).expand(bsz, output_len, src_len), attn * pointers[:,:,1].unsqueeze(-1))
        vocab_dist = vocab_dist.scatter_add(2, fact_extend_ids.unsqueeze(1).expand(bsz, output_len, fact_len), fact_attn * pointers[:,:,2].unsqueeze(-1))
        if persona_attn is not None:
            pers_len = pers_extend_ids.size(1)
            vocab_dist = vocab_dist.scatter_add(2, pers_extend_ids.unsqueeze(1).expand(bsz, output_len, pers_len), persona_attn * pointers[:,:,3].unsqueeze(-1))
        if topic_attn is not None:    
            bow_len = bow_ids.size(1)
            if persona_attn is not None:
                vocab_dist = vocab_dist.scatter_add(2, bow_ids.unsqueeze(1).expand(bsz, output_len, bow_len), topic_attn * pointers[:,:,4].unsqueeze(-1))
            else:
                vocab_dist = vocab_dist.scatter_add(2, bow_ids.unsqueeze(1).expand(bsz, output_len, bow_len), topic_attn * pointers[:,:,3].unsqueeze(-1))
        return vocab_dist

    def forward(self, src, dec, targets, fact, pers, pers_bow, src_extend_ids, tgt_extend_ids, fact_extend_ids, pers_extend_ids, bow_ids, max_ext_len):
        """
        run transformer
        :param src: source input
        :param src_len: source length
        :param dec: decoder input
        :param targets: target
        :return: dictionary of loss, reward, etc., output scores
        """
        return_dict = {}
        mask = (src != 0).float()
        fd1, fd2, fd3 = fact.size() # batch, num, len
        fact_mask = fact.data.eq(self.padding_idx)
        fact = fact.view(-1, fd3) # batch*num, len
        fact_extend_ids = fact_extend_ids.view(fd1, -1)

        contexts = self.encoder(src)  # batch, len, size
        fact_mask = fact_mask.view(fd1, -1)
        fact_contexts = self.fact_encoder(fact) # batch*num, len, size
        fact_contexts = fact_contexts.view(fd1, fd2*fd3, -1) # batch, num*len, size 

        fact_contexts = self.condition_context_attn(fact_contexts, contexts, mask)
        fact_contexts = self.bi_attn_transform(fact_contexts)

        persona_contexts = None
        persona_mask = None
        persona_topic_emb = None
        persona_word_emb = None
        if self.config.persona:
            if self.config.experience:
                pd1, pd2, pd3 = pers.size() # batch, num, len
                persona_mask = pers.data.eq(self.padding_idx)
                pers = pers.view(-1, pd3) # batch*num, len
                pers_extend_ids = pers_extend_ids.view(pd1, -1)
                persona_mask = persona_mask.view(pd1, -1)
                persona_contexts = self.persona_encoder(pers) # batch*num, len, size
                persona_contexts = persona_contexts.view(pd1, pd2*pd3, -1) # batch, num*len, size 

                persona_contexts = self.persona_condition_context_attn(persona_contexts, contexts, mask)
                persona_contexts = self.persona_bi_attn_transform(persona_contexts)

            if self.config.vae:
                bow_emb = self.encoder.embedding(bow_ids[0])
                persona_topic_emb, persona_word_emb, recon_batch, mu, logvar, reg = self.npm(pers_bow, bow_emb)
                return_dict["vae_loss"] = self.VAE_loss(recon_batch, pers_bow, mu, logvar) + reg

        models.transformer.init_state(self.decoder, src, contexts, self.decoder.num_layers)

        outputs, attn_weights, fact_attn, persona_attn, topic_attn, pointers = self.decoder(dec, contexts, fact_contexts, fact_mask, persona_contexts, persona_mask, persona_topic_emb, persona_word_emb) # [batch, len, size]

        scores = self.compute_score(outputs) # [batch, len, vocab]
        if pointers is not None:
            scores = self.pointer_network(scores, attn_weights, fact_attn, persona_attn, topic_attn, pointers, src_extend_ids, fact_extend_ids, pers_extend_ids, bow_ids, max_ext_len)
            scores = torch.log(scores.clamp(min=1e-8))
            return_dict["mle_loss"] = self.compute_loss(scores, tgt_extend_ids)
        else:
            scores = F.log_softmax(scores, dim=-1)
            return_dict["mle_loss"] = self.compute_loss(scores, targets)
        return return_dict, scores

    def sample(self, src, fact, pers, pers_bow, src_extend_ids, fact_extend_ids, pers_extend_ids, bow_ids, max_ext_len):
        """
        Greedy sampling for inference
        :param src: source input
        :param src_len: source length
        :return: sampled ids and attention alignment
        """
        bos = torch.ones(src.size(0)).long().fill_(utils.BOS).to(self.device)   # [batch]
        
        mask = (src != 0).float()
        fd1, fd2, fd3 = fact.size() # batch, num, len
        fact_mask = fact.data.eq(self.padding_idx)
        fact = fact.view(-1, fd3) # batch*num, len
        fact_extend_ids = fact_extend_ids.view(fd1, -1)

        contexts = self.encoder(src)  # batch, len, size
        fact_mask = fact_mask.view(fd1, -1)
        fact_contexts = self.fact_encoder(fact) # batch*num, len, size
        fact_contexts = fact_contexts.view(fd1, fd2*fd3, -1) # batch, num*len, size 

        fact_contexts = self.condition_context_attn(fact_contexts, contexts, mask)
        fact_contexts = self.bi_attn_transform(fact_contexts)

        persona_contexts = None
        persona_mask = None
        persona_topic_emb = None
        persona_word_emb = None
        if self.config.persona and pers is not None:
            if self.config.experience:
                pd1, pd2, pd3 = pers.size() # batch, num, len
                persona_mask = pers.data.eq(self.padding_idx)
                pers = pers.view(-1, pd3) # batch*num, len
                pers_extend_ids = pers_extend_ids.view(pd1, -1)
                persona_mask = persona_mask.view(pd1, -1)
                persona_contexts = self.persona_encoder(pers) # batch*num, len, size
                persona_contexts = persona_contexts.view(pd1, pd2*pd3, -1) # batch, num*len, size 

                persona_contexts = self.persona_condition_context_attn(persona_contexts, contexts, mask)
                persona_contexts = self.persona_bi_attn_transform(persona_contexts)

            if self.config.vae:
                bow_emb = self.encoder.embedding(bow_ids[0])
                persona_topic_emb, persona_word_emb, _, _, _, _ = self.npm(pers_bow, bow_emb)

        models.transformer.init_state(self.decoder, src, contexts, self.decoder.num_layers)

        # sequential generation
        inputs, outputs, attn_matrix = [bos], [], []
        for i in range(self.config.max_time_step):
            output, attn_weights, fact_attn, persona_attn, topic_attn, pointers = self.decoder(inputs[i].unsqueeze(1), contexts, fact_contexts, fact_mask, persona_contexts, persona_mask, persona_topic_emb, persona_word_emb, step=i) # [batch, len, size]
            output = self.compute_score(output)  # [batch, 1, size]
            if pointers is not None:
                output = self.pointer_network(output, attn_weights, fact_attn, persona_attn, topic_attn, pointers, src_extend_ids, fact_extend_ids, pers_extend_ids, bow_ids, max_ext_len)
            predicted = output.squeeze(1).max(1)[1]    # [batch]
            latest_tokens = [t if t < self.config.tgt_vocab_size else utils.UNK for t in predicted]
            latest_tokens = torch.LongTensor(latest_tokens).to(self.device) 
            inputs.append(latest_tokens)
            outputs.append(predicted)
            attn_matrix.append(attn_weights.squeeze(1)) #[batch, k_len]
        outputs = torch.stack(outputs)  # [batch, len]
        # select by the indices along the dimension of batch
        sample_ids = outputs.t().tolist()

        attn_matrix = torch.stack(attn_matrix)  # [batch, len, k_len]
        alignments = attn_matrix.max(2)[1].t().tolist() # [batch, len]
        
        return sample_ids, alignments

    def beam_sample(self, src, fact, pers, pers_bow, src_extend_ids, fact_extend_ids, pers_extend_ids, bow_ids, max_ext_len, beam_size=1, eval_=False):
        """
        beam search
        :param src: source input
        :param src_len: source length
        :param beam_size: beam size
        :param eval_: evaluation or not
        :return: prediction, attention max score and attention weights
        """
        batch_size = src.size(0)

        mask = (src != 0).float()
        fd1, fd2, fd3 = fact.size() # batch, num, len
        fact_mask = fact.data.eq(self.padding_idx)
        fact = fact.view(-1, fd3) # batch*num, len
        fact_extend_ids = fact_extend_ids.view(fd1, -1)

        contexts = self.encoder(src)  # batch, len, size
        fact_mask = fact_mask.view(fd1, -1)
        fact_contexts = self.fact_encoder(fact) # batch*num, len, size
        fact_contexts = fact_contexts.view(fd1, fd2*fd3, -1) # batch, num*len, size 

        fact_contexts = self.condition_context_attn(fact_contexts, contexts, mask)
        fact_contexts = self.bi_attn_transform(fact_contexts)

        persona_contexts = None
        persona_mask = None
        persona_topic_emb = None
        persona_word_emb = None
        if self.config.persona and pers is not None:
            if self.config.experience:
                pd1, pd2, pd3 = pers.size() # batch, num, len
                persona_mask = pers.data.eq(self.padding_idx)
                pers = pers.view(-1, pd3) # batch*num, len
                pers_extend_ids = pers_extend_ids.view(pd1, -1)
                persona_mask = persona_mask.view(pd1, -1)
                persona_contexts = self.persona_encoder(pers) # batch*num, len, size
                persona_contexts = persona_contexts.view(pd1, pd2*pd3, -1) # batch, num*len, size 

                persona_contexts = self.persona_condition_context_attn(persona_contexts, contexts, mask)
                persona_contexts = self.persona_bi_attn_transform(persona_contexts)

            if self.config.vae:
                bow_emb = self.encoder.embedding(bow_ids[0])
                persona_topic_emb, persona_word_emb, _, _, _, _ = self.npm(pers_bow, bow_emb)

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(batch_size, beam_size, -1)

        beam = [models.Beam(beam_size, n_best=1,
                            cuda=self.use_cuda, length_norm=self.config.length_norm)
                for __ in range(batch_size)]    # [batch, beam]

        contexts = tile(contexts, beam_size, 0) # [batch*beam, len, size]
        src = tile(src, beam_size, 0)   # [batch*beam, len]
        src_extend_ids = tile(src_extend_ids, beam_size, 0)

        fact_contexts = tile(fact_contexts, beam_size, 0) # [batch*beam, len, size]
        fact_mask = tile(fact_mask, beam_size, 0) # [batch*beam, len]
        fact_extend_ids = tile(fact_extend_ids, beam_size, 0)
        if persona_contexts is not None:
            persona_contexts = tile(persona_contexts, beam_size, 0) # [batch*beam, len, size]
            persona_mask = tile(persona_mask, beam_size, 0) # [batch*beam, len]
            pers_extend_ids = tile(pers_extend_ids, beam_size, 0)
        if self.config.vae:
            persona_topic_emb = tile(persona_topic_emb, beam_size, 0)
            persona_word_emb = tile(persona_word_emb, beam_size, 0)
        bow_ids = tile(bow_ids, beam_size, 0)

        # self.decoder.init_state(src, contexts)
        models.transformer.init_state(self.decoder, src, contexts, self.decoder.num_layers)

        # sequential generation
        for i in range(self.config.max_time_step):
            # finish beam search
            if all((b.done() for b in beam)):
                break

            inp = torch.stack([torch.LongTensor([t if t < self.config.tgt_vocab_size else utils.UNK for t in b.getCurrentState()]).to(self.device) for b in beam])
            inp = inp.view(-1, 1)   # [batch*beam, 1]

            output, attn, fact_attn, persona_attn, topic_attn, pointers = self.decoder(inp, contexts, fact_contexts, fact_mask, persona_contexts, persona_mask, persona_topic_emb, persona_word_emb, step=i) # [batch*beam, len, size]
            state = None
            output = self.compute_score(output)  # [batch*beam, 1, size]
            if pointers is not None:
                output = self.pointer_network(output, attn, fact_attn, persona_attn, topic_attn, pointers, src_extend_ids, fact_extend_ids, pers_extend_ids, bow_ids, max_ext_len)
                output = unbottle(torch.log(output.squeeze(1).clamp(min=1e-8)))
            else:
                output = unbottle(self.log_softmax(output.squeeze(1))) # [batch, beam, size]
            attn = unbottle(attn.squeeze(1))    # [batch, beam, k_len]

            select_indices_array = []
            # scan beams in a batch
            for j, b in enumerate(beam):
                # update each beam
                b.advance(output[j, :], attn[j, :]) # batch index
                select_indices_array.append(
                    b.getCurrentOrigin() + j * beam_size)
            select_indices = torch.cat(select_indices_array)
            self.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        allHyps, allScores, allAttn = [], [], []
        if eval_:
            allWeight = []

        for j in range(batch_size):
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            if eval_:
                weight = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
                if eval_:
                    weight.append(att)
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])
            if eval_:
                allWeight.append(weight[0])
        if eval_:
            return allHyps, allAttn, allWeight

        return allHyps, allAttn
