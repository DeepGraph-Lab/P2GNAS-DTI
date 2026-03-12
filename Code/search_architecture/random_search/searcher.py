import random

CONV_CHOICES = ['sage', 'graphconv', 'transformerconv', 'gatconv', 'gatv2conv']
NORM_CHOICES = ['graphnorm', 'instancenorm', 'layernorm', 'batchnorm', 'identity']
ACT_CHOICES = ['relu', 'tanh', 'leakyrelu', 'identity']


class RandomSearcher:

    def __init__(self, max_layers=5):

        self.max_layers = max_layers

    def sample(self):

        num_layers = random.randint(1, self.max_layers)

        layer_configs = []
        for _ in range(num_layers):
            config = {
                'conv': random.choice(CONV_CHOICES),
                'norm': random.choice(NORM_CHOICES),
                'act': random.choice(ACT_CHOICES)
            }
            layer_configs.append(config)

        return layer_configs