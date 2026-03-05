# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from inspect import signature
import torch
from torch import nn
import math
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):

        device = x.device
        half_dim = (self.dim // 2) + 1
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        emb = x[:, None] * emb[None, :]

        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        return emb[...,:self.dim]


class ConcatLinear_v1(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)


    def forward(self, t, x):

        return self._layer(x) 


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_bias.weight.data.fill_(0.0)

    def forward(self, t, x):

        return self._layer(x) + self._hyper_bias(t.view(1,1,1))
    
class ConcatLinear_te(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        # self._hyper_bias = nn.Linear(1, dim_out, bias=False)
       
        sinu_pos_emb = SinusoidalPosEmb(dim_out)
        self._time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim_out, dim_out),
            nn.GELU(),
            nn.Linear(dim_out, dim_out)
        )

        # 初始化所有 Linear 层的权重为 0
        for layer in self._time_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0.0)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, t, x):

        return self._layer(x) + self._time_mlp(t.view(1,1))

class ConcatLinearNorm(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_bias.weight.data.fill_(0.0)
        self.norm = nn.LayerNorm(dim_out, eps=1e-6)

    def forward(self, t, x):
        return self.norm(self._layer(x) + self._hyper_bias(t.view(1,1,1)))


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1,1,1))) \
               + self._hyper_bias(t.view(1,1,1))


class DiffEqWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, t, y):
        if len(signature(self.module.forward).parameters) == 1:
            return self.module(y)
        if len(signature(self.module.forward).parameters) == 2:
            return self.module(t, y)
        raise ValueError("Differential equation needs to either take (t, y) or (y,) as input.")

    def __repr__(self):
        return self.module.__repr__()


def diffeq_wrapper(layer):
    return DiffEqWrapper(layer)


class SequentialDiffEq(nn.Module):
    """A container for a sequential chain of layers. Supports both regular and diffeq layers.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList([diffeq_wrapper(layer) for layer in layers])

    def forward(self, t, x):
        for layer in self.layers:
            x = layer(t, x)
        return x


class TimeDependentSwish(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.beta = nn.Sequential(
            nn.Linear(1, min(64, dim * 4)),
            nn.Softplus(),
            nn.Linear(min(64, dim * 4), dim),
            nn.Softplus(),
        )

    def forward(self, t, x):
        beta = self.beta(t.reshape(-1, 1))
        return x * torch.sigmoid_(x * beta)





ACTFNS = {
    "softplus": (lambda dim: nn.Softplus()),
    "tanh": (lambda dim: nn.Tanh()),
    "swish": TimeDependentSwish,
    "relu": (lambda dim: nn.ReLU()),
    'leakyrelu': (lambda dim: nn.LeakyReLU()),
    'sigmoid': (lambda dim: nn.Sigmoid()),
}

LAYERTYPES = {
    "concatsquash": ConcatSquashLinear,
    "concat": ConcatLinear_v2,
    "concatlinear": ConcatLinear_v2,
    "concatnorm": ConcatLinearNorm,
    "concat_v1": ConcatLinear_v1,
    "concat_te":ConcatLinear_te,
}


def build_fc_odefunc(dim=2, hidden_dims=[64, 64, 64], out_dim=None, nonzero_dim=None, actfn="softplus",
                     layer_type="concat",
                     zero_init=True, actfirst=False):
    assert layer_type in LAYERTYPES, f"layer_type must be one of {LAYERTYPES.keys()} but was given {layer_type}"
    layer_fn = LAYERTYPES[layer_type]
    if layer_type == "concatlinear":
        hidden_dims = None

    nonzero_dim = dim if nonzero_dim is None else nonzero_dim
    out_dim = out_dim or hidden_dims[-1]
    if hidden_dims:
        dims = [dim] + list(hidden_dims)
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(layer_fn(d_in, d_out))
            layers.append(ACTFNS[actfn](d_out))
        layers.append(layer_fn(hidden_dims[-1], out_dim))
        
    else:
        layers = [layer_fn(dim, out_dim), ACTFNS[actfn](out_dim)]

    if actfirst and len(layers) > 1:
        layers = layers[1:]

    if nonzero_dim < dim:
        # zero out weights for auxiliary inputs.
        layers[0]._layer.weight.data[:, nonzero_dim:].fill_(0)

    if zero_init:
        for m in layers[-2].modules():
            if isinstance(m, nn.Linear):
                m.weight.data.fill_(0)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    return SequentialDiffEq(*layers)

