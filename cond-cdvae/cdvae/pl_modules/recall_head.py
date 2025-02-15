import hydra
import torch.nn as nn

from cdvae.pl_modules.basic_blocks import build_mlp


class PropRecall(nn.Module):
    def __init__(self, prop_keys: list, types: dict):
        super().__init__()
        self.prop_keys = prop_keys
        self.prop_model_list = nn.ModuleList(
            hydra.utils.instantiate(types[prop_key]) for prop_key in prop_keys
        )

    def forward(self, batch):
        return {
            prop_key: sub_model(batch[prop_key])
            for prop_key, sub_model in zip(self.prop_keys, self.prop_model_list)
        }


class RecallScalar(nn.Module):
    def __init__(self, prop_name: str, *args, **kwargs):
        super().__init__()
        self.prop_name = prop_name
        self.mlp = build_mlp(*args, **kwargs)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        y = self.mlp(x)
        return y
