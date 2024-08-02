import os
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = os.getenv("DEVICE", "cpu")


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def parameters(self):
        for _, param in self.named_params(self):
            yield param

    def named_parameters(self):
        for name, param in self.named_params(self):
            yield name, param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=""):
        if memo is None:
            memo = set()

        if hasattr(curr_module, "named_leaves"):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ("." if prefix else "") + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ("." if prefix else "") + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ("." if prefix else "") + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(
        self, lr_inner, first_order=False, source_params=None, detach=False
    ):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = grad.detach().data.clone().requires_grad_(True)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = grad.detach().data.clone().requires_grad_(True)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if "." in name:
            n = name.split(".")
            module_name = n[0]
            rest = ".".join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = param.data.clone().requires_grad_(True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer("weight", ignore.weight.data.clone().to(kwargs.get("device", DEVICE)).requires_grad_(True))
        if ignore.bias is not None:
            self.register_buffer("bias", ignore.bias.data.clone().to(kwargs.get("device", DEVICE)).requires_grad_(True))
        else:
            self.bias = None

    @property
    def in_features(self):
        return self.weight.shape[1]

    @property
    def out_features(self):
        return self.weight.shape[0]

    def named_leaves(self):
        if self.bias is None:
            return [("weight", self.weight)]
        return [("weight", self.weight), ("bias", self.bias)]

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class MetaParameter(MetaModule):
    def __init__(self, param, device=DEVICE):
        super().__init__()
        self.register_buffer("param", param.data.clone().to(device).requires_grad_(True))

    def named_leaves(self):
        return [("param", self.param)]

    def forward(self):
        return self.param


class MetaBatchNorm1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm1d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer("weight", ignore.weight.data.clone().to(kwargs.get("device", DEVICE)).requires_grad_(True))
            self.register_buffer("bias", ignore.bias.data.clone().to(kwargs.get("device", DEVICE)).requires_grad_(True))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(self.num_features))
            self.register_buffer("running_var", torch.ones(self.num_features))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)

    def forward(self, x):
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )

    def named_leaves(self):
        named_leaves = [("weight", self.weight), ("bias", self.bias)]
        if self.track_running_stats:
            named_leaves += [
                ("running_mean", self.running_mean),
                ("running_var", self.running_var),
            ]
        return named_leaves
