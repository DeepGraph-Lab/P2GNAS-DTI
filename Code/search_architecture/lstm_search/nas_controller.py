
import torch
from torch import nn
from torch.distributions import Categorical


CONV_CHOICES = ['sage', 'graphconv', 'transformerconv', 'gatconv', 'gatv2conv']
NORM_CHOICES = ['graphnorm', 'instancenorm', 'layernorm', 'batchnorm', 'identity']
ACT_CHOICES = ['relu', 'tanh', 'leakyrelu', 'identity']


class NAS_Controller(nn.Module):
    def __init__(self, max_layers=5, hidden_dim=32):
        super().__init__()
        self.max_layers = max_layers
        self.hidden_dim = hidden_dim

        self.conv_map = {name: i for i, name in enumerate(CONV_CHOICES)}
        self.norm_map = {name: i for i, name in enumerate(NORM_CHOICES)}
        self.act_map = {name: i for i, name in enumerate(ACT_CHOICES)}

        self.conv_embedding = nn.Embedding(len(CONV_CHOICES), hidden_dim)
        self.norm_embedding = nn.Embedding(len(NORM_CHOICES), hidden_dim)
        self.act_embedding = nn.Embedding(len(ACT_CHOICES), hidden_dim)

        self.lstm_cell = nn.LSTMCell(input_size=3 * hidden_dim, hidden_size=hidden_dim)

        self.conv_head = nn.Linear(hidden_dim, len(CONV_CHOICES))
        self.norm_head = nn.Linear(hidden_dim, len(NORM_CHOICES))
        self.act_head = nn.Linear(hidden_dim, len(ACT_CHOICES))

        self.stop_head = nn.Linear(hidden_dim, 2)

        self.start_token = nn.Parameter(torch.randn(1, 3 * hidden_dim), requires_grad=True)

    def sample(self):

        device = self.start_token.device
        input_token = self.start_token
        h, c = (torch.zeros(1, self.hidden_dim).to(device),
                torch.zeros(1, self.hidden_dim).to(device))

        total_log_prob = 0
        layer_configs = []

        for layer_idx in range(self.max_layers):
            h, c = self.lstm_cell(input_token, (h, c))

            conv_logits = self.conv_head(h)
            norm_logits = self.norm_head(h)
            act_logits = self.act_head(h)

            conv_dist = Categorical(logits=conv_logits)
            norm_dist = Categorical(logits=norm_logits)
            act_dist = Categorical(logits=act_logits)

            conv_idx = conv_dist.sample()
            norm_idx = norm_dist.sample()
            act_idx = act_dist.sample()

            config = {
                'conv': CONV_CHOICES[conv_idx.item()],
                'norm': NORM_CHOICES[norm_idx.item()],
                'act': ACT_CHOICES[act_idx.item()]
            }
            layer_configs.append(config)

            log_prob_layer = (conv_dist.log_prob(conv_idx) +
                              norm_dist.log_prob(norm_idx) +
                              act_dist.log_prob(act_idx))
            total_log_prob += log_prob_layer

            conv_emb = self.conv_embedding(conv_idx)
            norm_emb = self.norm_embedding(norm_idx)
            act_emb = self.act_embedding(act_idx)
            input_token = torch.cat([conv_emb, norm_emb, act_emb], dim=-1)

            stop_logits = self.stop_head(h)
            stop_dist = Categorical(logits=stop_logits)
            stop_decision = stop_dist.sample()

            total_log_prob += stop_dist.log_prob(stop_decision)

            if stop_decision == 1:
                break

        return layer_configs, total_log_prob