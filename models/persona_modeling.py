import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class NTM(nn.Module):
    def __init__(self, config, hidden_dim=512, l1_strength=0.001):
        super(NTM, self).__init__()
        self.input_dim = config.bow_vocab_size
        self.topic_num = config.topic_num
        topic_num = config.topic_num
        self.fc11 = nn.Linear(self.input_dim, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, topic_num)
        self.fc22 = nn.Linear(hidden_dim, topic_num)
        self.fcs = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.fcg1 = nn.Linear(topic_num, topic_num)
        self.fcg2 = nn.Linear(topic_num, topic_num)
        self.fcg3 = nn.Linear(topic_num, topic_num)
        self.fcg4 = nn.Linear(topic_num, topic_num)
        self.fcd1 = nn.Linear(topic_num, self.input_dim)

    def encode(self, x):
        e1 = F.relu(self.fc11(x))
        e1 = F.relu(self.fc12(e1))
        e1 = e1.add(self.fcs(x))
        return self.fc21(e1), self.fc22(e1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def generate(self, h):
        g1 = torch.tanh(self.fcg1(h))
        g1 = torch.tanh(self.fcg2(g1))
        g1 = torch.tanh(self.fcg3(g1))
        g1 = torch.tanh(self.fcg4(g1))
        g1 = g1.add(h)
        return g1

    def decode(self, z):
        d1 = F.softmax(self.fcd1(z), dim=1)
        return d1

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        g = self.generate(z)
        return z, g, self.decode(g), mu, logvar

    def print_topic_words(self, vocab_dic, fn, n_top_words=10):
        beta_exp = self.fcd1.weight.data.cpu().numpy().T
        print("Writing to %s" % fn)
        fw = open(fn, 'w')
        for k, beta_k in enumerate(beta_exp):
            topic_words = [vocab_dic.idxToLabel[vocab_dic.bowToIdx[w_id]] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
            print('Topic {}: {}'.format(k, ' '.join(topic_words)))
            fw.write('{}\n'.format(' '.join(topic_words)))
        fw.close()


class Persona_Fusion(nn.Module):
    def __init__(self, config):
        super(Persona_Fusion, self).__init__()
        self.linear_layer1 = nn.Linear(config.topic_num, config.hidden_size)
        self.linear_layer2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.linear_layer3 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, context, persona_embedding):
        bsz, ctx_len, emb = context.size()
        persona_embedding = self.linear_layer1(persona_embedding)
        context = self.linear_layer2(context)
        persona_embedding = persona_embedding.unsqueeze(1)
        persona_context = self.tanh(context + persona_embedding)
        return self.linear_layer3(persona_context)



class NPM(nn.Module):
    def __init__(self, config, hidden_dim=512):
        super(NPM, self).__init__()
        self.input_dim = config.bow_vocab_size
        self.topic_num = config.topic_num
        topic_num = config.topic_num
        self.fc11 = nn.Linear(self.input_dim, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, hidden_dim)
        self.fc22 = nn.Linear(hidden_dim, hidden_dim)
        self.fcs = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.dropout1 = nn.Dropout(config.dropout)

        self.ft = nn.Linear(hidden_dim, topic_num)
        self.E = nn.Parameter(data=torch.Tensor(topic_num, hidden_dim))
        #self.F = nn.Parameter(data=torch.Tensor(self.input_dim, hidden_dim))
    
    def encode(self, x):
        e1 = F.relu(self.fc11(x))
        e1 = F.relu(self.fc12(e1))
        #e1 = e1.add(self.fcs(x))
        e1 = self.dropout1(e1)
        return self.fc21(e1), self.fc22(e1)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, x, bow_emb, ortho_reg=0.1):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        t = F.softmax(self.ft(z), dim=-1) # batch, topic_num
        self.F = bow_emb
        self.beta = F.softmax(torch.mm(self.E, self.F.t()), dim=-1) # topic_num, vocab_size
        x_ = torch.matmul(t.unsqueeze(1), self.beta).squeeze(1) # batch, vocab_size

        persona_topic_emb = torch.mul(t.unsqueeze(-1), self.E)
        persona_word_emb = torch.mul(x_.unsqueeze(-1), self.F)
        return persona_topic_emb, persona_word_emb, x_, mu, logvar, ortho_reg * self._ortho_regularizer()
    
    def _ortho_regularizer(self):
        return torch.norm(torch.matmul(self.E, self.E.t()) - torch.eye(self.topic_num).to(self.E.device))
    
    def print_topic_words(self, vocab_dic, fn, n_top_words=50):
        beta_exp = self.beta.data.cpu().numpy()
        print("Writing to %s" % fn)
        fw = open(fn, 'w')
        for k, beta_k in enumerate(beta_exp):
            topic_words = [vocab_dic.idxToLabel[vocab_dic.bowToIdx[w_id]] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
            print('Topic {}: {}'.format(k, ' '.join(topic_words[:10])))
            fw.write('{}\n'.format(' '.join(topic_words)))
        fw.close()