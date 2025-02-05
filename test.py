from matplotlib.font_manager import weight_dict

from tcv import tcv

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self._input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self._fc_mean  = nn.Linear(hidden_dim, latent_dim)
        self._fc_var   = nn.Linear (hidden_dim, latent_dim)

    def forward(self, x):
        output_  = self._input(x)
        mean_    = self._fc_mean(output_)
        log_var_ = self._fc_var(output_)
        return mean_, log_var_

enc = Encoder(1000, 1200, 4)
x = np.random.normal(1, size=1000)
t = torch.tensor(x, dtype=torch.float32).reshape(1, 1000)
out = enc(t)

for k,v in tcv.build_graph(out[0], params=dict(enc.named_parameters()))[0].items():
    print(v)

g = tcv.weight_matrix(enc, '_input.2')
plt.savefig("w_example.png")
