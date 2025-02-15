import torch.nn as nn


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim, dropout=0., *args, **kwargs):
    if in_dim is None:
        mods = [nn.LazyLinear(hidden_dim), nn.ReLU()]
    else:
        mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        mods += [nn.Dropout(dropout)]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)
