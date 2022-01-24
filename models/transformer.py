import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models
import utils
from models import rnn


MAX_SIZE = 200


class PositionalEncoding(nn.Module):
    """ positional encoding """

    def __init__(self, dropout, dim, max_len=200):
        """
        initialization of required variables and functions
        :param dropout: dropout probability
        :param dim: hidden size
        :param max_len: maximum length
        """
        # positional encoding initialization
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        # term to divide
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        # sinusoidal positional encoding
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """
        create positional encoding
        :param emb: word embedding
        :param step: step for decoding in inference
        :return: positional encoding representation
        """
        # division of size
        emb = emb * math.sqrt(self.dim)
        if step is None:
            # residual connection
            emb = emb + self.pe[:,:emb.size(1)]   # [batch, len, size]
        else:
            # step for inference
            emb = emb + self.pe[:,step]   # [batch, len, size]
        emb = self.dropout(emb)
        return emb


class PositionwiseFeedForward(nn.Module):
    """ Point-wise Feed-Forward NN, FFN, in fact 1-d convolution """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        initialization of required functions
        :param d_model: model size
        :param d_ff: intermediate size
        :param dropout: dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        run FFN
        :param x: input
        :return: output
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        # with residual connection
        return output + x


class TransformerEncoderLayer(nn.Module):
    """ Transformer encoder layer """

    def __init__(self, config):
        """
        initialization of required variables and functions
        :param config: configuration
        """
        super(TransformerEncoderLayer, self).__init__()
        self.config = config
        # self attention
        self.self_attn = models.Multihead_Attention(
            model_dim=config.hidden_size, head_count=config.heads, dropout=config.dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=config.hidden_size, d_ff=config.d_ff, dropout=config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs, mask):
        """
        run transformer encoder layer
        :param inputs: inputs
        :param mask: mask
        :return: output
        """
        # self attention
        input_norm = self.layer_norm(inputs)  # [batch, len, size]
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                        mask=mask)  # [batch, len, size]
        out = self.dropout(context) + inputs    # [batch, len, size]
        # FFN
        return self.feed_forward(out)   # [batch, len, size]


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask=None):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        if mask is not None:
            att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1) # bsz, input_len, memory_len
        output_one = torch.bmm(weight_one, memory) # bsz, memory_len, hidden_size
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len) # bsz, 1, input_len
        output_two = torch.bmm(weight_two, input) # bsz, 1, hidden_size

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)


class TransformerEncoder(nn.Module):
    """ Transformer encoder """

    def __init__(self, config, padding_idx=0):
        """
        initialization of required variables and functions
        :param config: configuration
        :param padding_idx: index for padding in the dictionary
        """
        super(TransformerEncoder, self).__init__()

        self.config = config
        self.num_layers = config.enc_num_layers

        # HACK: 512 for word embeddings, 512 for condition embeddings
        self.embedding = nn.Embedding(config.src_vocab_size, config.emb_size,
                                      padding_idx=padding_idx)
        # positional encoding
        self.position_embedding = PositionalEncoding(
                config.dropout, config.emb_size)

        # transformer
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(config)
             for _ in range(config.enc_num_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.padding_idx = padding_idx

    def forward(self, src):
        """
        run transformer encoder
        :param src: source input
        :return: output
        """
        embed = self.embedding(src)

        out = self.position_embedding(embed)    # [batch, len, size]
 
        src_words = src  # [batch, len]
        src_batch, src_len = src_words.size()
        padding_idx = self.padding_idx
        mask = src_words.data.eq(padding_idx).unsqueeze(1).expand(src_batch, src_len, src_len)    # [batch, len, len]

        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)  # [batch, len, size]

        return out


# Decoder
class TransformerDecoderLayer(nn.Module):
    """ Transformer decoder layer """

    def __init__(self, config):
        """
        initialization for required variables and functions
        :param config: configuration
        """
        super(TransformerDecoderLayer, self).__init__()
        self.config = config
        # self attention
        self.self_attn = models.Multihead_Attention(
            model_dim=config.hidden_size, head_count=config.heads, dropout=config.dropout)

        self.context_attn = models.Multihead_Attention(
            model_dim=config.hidden_size, head_count=config.heads, dropout=config.dropout)
        self.fact_attn = models.Multihead_Attention(
            model_dim=config.hidden_size, head_count=config.heads, dropout=config.dropout)
        
        self.feed_forward = PositionwiseFeedForward(
            config.hidden_size, config.d_ff, config.dropout)
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        self.dropout = config.dropout
        self.drop = nn.Dropout(config.dropout)
        self.drop_f = nn.Dropout(config.dropout)

        if config.persona:
            self.persona_attn = models.Multihead_Attention(
                model_dim=config.hidden_size, head_count=config.heads, dropout=config.dropout)
            self.topic_attn = models.Multihead_Attention(
                model_dim=config.hidden_size, head_count=config.heads, dropout=config.dropout)
            self.drop_p = nn.Dropout(config.dropout)
            self.drop_t = nn.Dropout(config.dropout)
            self.layer_norm_4 = nn.LayerNorm(config.hidden_size, eps=1e-6)
            self.layer_norm_5 = nn.LayerNorm(config.hidden_size, eps=1e-6)

        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)


    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                fact_memory, fact_pad_mask, persona_memory=None, persona_pad_mask=None, persona_topic_emb=None,
                layer_cache=None, step=None):
        """
        run transformer decoder layer
        :param inputs: inputs
        :param memory_bank: source representations
        :param src_pad_mask: source padding mask
        :param tgt_pad_mask: target padding mask
        :param layer_cache: layer cache for decoding in inference stage
        :param step: step for decoding in inference stage
        :return: output, attention weights and input norm
        """
        dec_mask = torch.gt(tgt_pad_mask
                            + self.mask[:, :tgt_pad_mask.size(1), :tgt_pad_mask.size(1)], 0)

        # self attention
        input_norm = self.layer_norm_1(inputs)
        query, _ = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         type="self")   # [batch, q_len, size]
        # residual connection
        query = self.drop(query) + inputs   # [batch, q_len, size]

        # context attention
        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                          mask=src_pad_mask,
                                          layer_cache=layer_cache,
                                          type="context",
                                          Bernoulli=self.config.Bernoulli)  # [batch, q_len, size]
        mid = self.drop(mid) + query
        
        # fact attention
        mid_norm = self.layer_norm_3(mid)
        fact_mid, fact_attn = self.fact_attn(fact_memory, fact_memory, mid_norm,
                                          mask=fact_pad_mask,
                                          layer_cache=layer_cache,
                                          type="fact",
                                          Bernoulli=self.config.Bernoulli)  # [batch, q_len, size]
        fact_mid = self.drop_f(fact_mid) + mid
        ff_mid = fact_mid
        #ff_mid = torch.mean(torch.stack([self.drop(mid), self.drop_f(fact_mid), self.drop_p(persona_mid), query], dim=2), dim=2)
        
        # persona attention
        if self.config.persona:
            if self.config.experience:
                ff_mid_norm = self.layer_norm_4(ff_mid)
                persona_mid, persona_attn = self.persona_attn(persona_memory, persona_memory, ff_mid_norm,
                                            mask=persona_pad_mask,
                                            layer_cache=layer_cache,
                                            type="persona",
                                            Bernoulli=self.config.Bernoulli)  # [batch, q_len, size]
                persona_mid = self.drop_p(persona_mid) + ff_mid
                ff_mid = persona_mid
            if self.config.vae:
                ff_mid_norm = self.layer_norm_5(ff_mid)
                topic_mid, topic_attn = self.topic_attn(persona_topic_emb, persona_topic_emb, ff_mid_norm,
                                            layer_cache=layer_cache,
                                            type="topic",
                                            Bernoulli=self.config.Bernoulli)  # [batch, q_len, size]
                topic_mid = self.drop_t(topic_mid) + ff_mid
                ff_mid = topic_mid

        output = self.feed_forward(ff_mid)  # [batch, q_len, size]
        return output, attn#, mid, fact_attn, fact_mid, persona_attn, persona_mid

    def _get_attn_subsequent_mask(self, size):
        """
        get mask for target
        :param size: max size
        :return: target mask
        """
        attn_shape = (1, size, size)    # [1, size, size]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    """ Transformer decoder """

    def __init__(self, config, tgt_embedding=None, padding_idx=0):
        """
        initialization for required variables and functions
        :param config: configuration
        :param tgt_embedding: target embedding
        :param padding_idx: padding index
        """
        super(TransformerDecoder, self).__init__()
        self.config = config
        self.num_layers = config.dec_num_layers
        if tgt_embedding:
            self.embedding = tgt_embedding
        else:
            self.embedding = nn.Embedding(config.tgt_vocab_size, config.emb_size,
                                          padding_idx=padding_idx)
        if config.positional:
            self.position_embedding = PositionalEncoding(
                config.dropout, config.emb_size)
        else:
            self.rnn = nn.LSTMCell(config.emb_size, config.hidden_size)

        self.padding_idx = padding_idx
        # state to store elements, including source and layer cache
        self.state = {}
        # transformer decoder
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(config)
             for _ in range(config.dec_num_layers)])
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        if config.pointer:
            pointer_num = 3
            self.question_attention = models.External_Attention(config.hidden_size)
            self.fact_attention = models.External_Attention(config.hidden_size)
            if config.experience:
                pointer_num += 1
                self.persona_attention = models.External_Attention(config.hidden_size)
            if config.vae:
                pointer_num += 1
                self.topic_attention = models.External_Attention(config.hidden_size)
            self.linear = nn.Linear(config.hidden_size * pointer_num, pointer_num)
            
            
    def forward(self, tgt, memory_bank, fact_memory, fact_mask, persona_memory=None, persona_mask=None, persona_topic_emb=None, persona_word_emb=None, state=None, step=None):
        """
        run transformer decoder
        :param tgt: target input
        :param memory_bank: source representations
        :param state: state
        :param step: step for inference
        :return: output, attention weights and state
        """
        src = self.state["src"]
        src_words = src  # [batch, src_len]
        tgt_words = tgt  # [batch, tgt_len]
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        emb = self.embedding(tgt)   # [batch, tgt_len, size]
        emb = self.position_embedding(emb, step=step)

        output = emb   # [batch, tgt_len, size]
        src_memory_bank = memory_bank   # [batch, src_len, size]

        padding_idx = self.padding_idx
        # source padding mask
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, src_len)    # [batch, tgt_len, src_len]
        # target padding mask
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)    # [batch, tgt_len, tgt_len]
        # fact padding mask
        fact_len = fact_mask.size()[-1]
        fact_pad_mask = fact_mask.unsqueeze(1).expand(src_batch, tgt_len, fact_len)
        # persona padding mask
        persona_pad_mask = None
        if self.config.experience:
            persona_len = persona_mask.size()[-1]
            persona_pad_mask = persona_mask.unsqueeze(1).expand(src_batch, tgt_len, persona_len)

        # run transformer decoder layers
        for i in range(self.num_layers):
            output, attn = self.transformer_layers[i](
                output, src_memory_bank,
                src_pad_mask, tgt_pad_mask,
                fact_memory, fact_pad_mask,
                persona_memory, persona_pad_mask, persona_topic_emb,
                layer_cache=self.state["cache"]["layer_{}".format(i)],
                step=step)#, question_context, fact_attn, fact_context, persona_attn, persona_context
        
        output = self.layer_norm(output)    # [batch, tgt_len, size]

        if self.config.pointer:
            question_context, attn = self.question_attention(src_memory_bank, src_memory_bank, output, src_pad_mask)
            fact_context, fact_attn = self.fact_attention(fact_memory, fact_memory, output, fact_pad_mask)
            if self.config.persona:
                if self.config.vae and self.config.experience:
                    persona_context, persona_attn = self.persona_attention(persona_memory, persona_memory, output, persona_pad_mask)
                    topic_context, topic_attn = self.topic_attention(persona_word_emb, persona_word_emb, output)
                    pointers = F.softmax(self.linear(torch.cat([output, question_context, fact_context, persona_context, topic_context], dim=-1)), dim=-1)
                    return output, attn, fact_attn, persona_attn, topic_attn, pointers
                elif self.config.vae and not self.config.experience:
                    topic_context, topic_attn = self.topic_attention(persona_word_emb, persona_word_emb, output)
                    pointers = F.softmax(self.linear(torch.cat([output, question_context, fact_context, topic_context], dim=-1)), dim=-1)
                    return output, attn, fact_attn, None, topic_attn, pointers
                else:
                    persona_context, persona_attn = self.persona_attention(persona_memory, persona_memory, output, persona_pad_mask)
                    pointers = F.softmax(self.linear(torch.cat([output, question_context, fact_context, persona_context], dim=-1)), dim=-1)
                    return output, attn, fact_attn, persona_attn, None, pointers
            else:
                pointers = F.softmax(self.linear(torch.cat([output, question_context, fact_context], dim=-1)), dim=-1)
                return output, attn, fact_attn, None, None, pointers
        else:
            return output, attn, None, None, None, None
            # [batch, tgt_len, size], [batch, tgt_len, src_len]
    
    def map_state(self, fn):
        """
        state mapping
        :param fn: function
        :return: none
        """
        def _recursive_map(struct, batch_dim=0):
            """
            recursive mapping
            :param struct: object for mapping
            :param batch_dim: batch dimension
            :return: none
            """
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)
        # self.state["src"] = fn(self.state["src"], 1)
        # layer cache mapping
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])


def init_state(self, src, memory_bank, num_layers):
    """
    state initialization, to replace the one in the transformer decoder
    :param self: self
    :param src: source input
    :param memory_bank: source representations
    :param num_layers: number of layers
    :return: none
    """
    self.state = {}
    self.state["src"] = src
    self.state["cache"] = {}

    # device for multi-gpus
    device = str(memory_bank.device)
    # print(device)

    memory_keys = "memory_keys_" + device
    memory_values = "memory_values_" + device
    self_keys = "self_keys_" + device
    # print(self_keys)
    self_values = "self_values_" + device
    fact_keys = "fact_keys_" + device
    fact_values = "fact_values_" + device
    persona_keys = "persona_keys_" + device
    persona_values = "persona_values_" + device
    topic_keys = "topic_keys_" + device
    topic_values = "topic_values_" + device

    # build layer cache for each layer
    for l in range(num_layers):
        layer_cache = {
            memory_keys: None,
            memory_values: None,
            self_keys: None,
            self_values: None,
            fact_keys: None,
            fact_values: None,
            persona_keys: None,
            persona_values: None,
            topic_keys: None,
            topic_values: None
        }
        # store in the cache in state
        self.state["cache"]["layer_{}".format(l)] = layer_cache