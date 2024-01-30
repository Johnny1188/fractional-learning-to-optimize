import torch
import torch.nn as nn
import torch.nn.functional as F

from fl2o.optimizee_modules import MetaModule, MetaLinear, MetaParameter


class Optee(nn.Module):
    """
    ...
    """
    def __init__(
        self,
    ):
        super().__init__()
        raise NotImplementedError

    def forward(self):
        """ returns metrics """
        raise NotImplementedError


class MLPOptee(MetaModule):
    def __init__(
        self,
        layer_sizes=[20],
        inp_size=28 * 28,
        out_size=10,
        act_fn=nn.ReLU(),
        out_act_fn=None,
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.inp_size = inp_size
        self.out_size = out_size
        self.act_fn = act_fn
        self.out_act_fn = out_act_fn

        ### init layers
        layers = []
        for i, layer_size in enumerate(layer_sizes):
            layers.append(MetaLinear(inp_size, layer_size))
            layers.append(act_fn)
            inp_size = layer_size
        layers.append(MetaLinear(inp_size, out_size))
        if out_act_fn is not None:
            layers.append(out_act_fn)
        self.layers = nn.Sequential(*layers)

    def get_deepcopy(self):
        optee_copy = MLPOptee(
            layer_sizes=self.layer_sizes,
            inp_size=self.inp_size,
            out_size=self.out_size,
            act_fn=self.act_fn,
            out_act_fn=self.out_act_fn,
        )
        my_params = self.all_named_parameters(to_dict=True)
        for n, p in optee_copy.all_named_parameters():
            p.data = my_params[n].data.detach().clone()
        return optee_copy

    def all_named_parameters(self, to_dict=False):
        if to_dict:
            return {k: v for k, v in self.named_parameters()}
        else:
            return [(k, v) for k, v in self.named_parameters()]

    def forward_w_params(self, params, params_for, task=None):
        ### allows to call forward() with a different set of params
        ### - first dim of params should be the number of params to run with
        x = task["x"].view(-1, self.inp_size)
        for l_idx, l in enumerate(self.layers):
            if "linear" in l.__class__.__name__.lower():
                if params_for == f"layers.{l_idx}.weight":
                    if params.ndim == 2:
                        x = x @ params.T + l.bias
                    elif  params.ndim == 3:
                        x = (params @ x.T).permute(2, 0, 1) + l.bias # TODO: check that the bias gets correct grad
                elif params_for == f"layers.{l_idx}.bias":
                    x = (x @ l.weight.T).unsqueeze(1).expand(-1, params.shape[0], -1) + params
                else:
                    x = l(x)
            else:
                x = l(x) # activation
        return x

    def forward(self, task):
        x = task["x"].view(-1, self.inp_size)
        x = self.layers(x)
        return x


def get_mlp_optee_mirror(optee): # make a deepcopy but using standard nn.Linear instead of MetaLinear
    optee_copy = optee.get_deepcopy()
    for l_idx in range(len(optee.layers)):
        l = optee.layers[l_idx]
        if l.__class__.__name__ == "MetaLinear":
            optee_copy.layers[l_idx] = nn.Linear(
                in_features=l.in_features,
                out_features=l.out_features,
                bias=l.bias is not None,
            )
            optee_copy.layers[l_idx].weight.data = l.weight.data.detach().clone()
            if l.bias is not None:
                optee_copy.layers[l_idx].bias.data = l.bias.data.detach().clone()
    return optee_copy


class CustomParams(MetaModule):
    def __init__(self, dim, init_params="zeros", param_func=None):
        super().__init__()

        self.dim = dim
        self.init_params = init_params
        self._params = None
        self.param_func = param_func
        self._reset_params()

    def _reset_params(self):
        if hasattr(self.init_params, "__call__"):
            self._params = self.init_params(dim=self.dim)
        else:
            if self.init_params == "zeros":
                self._params = MetaParameter(torch.zeros(self.dim))
            elif self.init_params == "ones":
                self._params = MetaParameter(torch.ones(self.dim))
            elif self.init_params == "randn":
                self._params = MetaParameter(torch.randn(self.dim))
            elif self.init_params == "rand":
                self._params = MetaParameter(torch.rand(self.dim))
            elif type(self.init_params) in [int, float]:
                self._params = MetaParameter(torch.ones(self.dim) * self.init_params)
            elif type(self.init_params) == torch.Tensor:
                self._params = MetaParameter(self.init_params)
            else:
                raise NotImplementedError

    @property
    def params(self):
        if self._params.__class__.__name__ == "MetaParameter":
            return self._params.param
        else:
            return self._params
    
    @params.setter
    def params(self, value):
        if self._params.__class__.__name__ == "MetaParameter":
            self._params.param = value
        else:
            self._params = value

    def forward_w_params(self, params, params_for, task=None):
        assert params_for == "params", "CustomParams only has one param"
        ### allows to call forward() with a different set of params
        if self.param_func is not None:
            assert task is not None
            return self.param_func(task=task, params=params)
        else:
            return params

    def get_deepcopy(self):
        return CustomParams(
            dim=self.dim,
            init_params=self.params.detach().clone(),
            param_func=self.param_func,
        )

    def all_named_parameters(self):
        return [("params", self.params)]

    def forward(self, task=None):
        if self.param_func is not None:
            assert task is not None
            return self.param_func(task=task, params=self.params)
        else:
            return self.params


# class CustomParams(nn.Module):
#     def __init__(self, dim, init_params="zeros", param_func=None):
#         super().__init__()

#         self.params = nn.Parameter(torch.empty(dim))
#         self.param_func = param_func
#         self._reset_params(init_params)

#     def _reset_params(self, init_params):
#         if init_params == "zeros":
#             self.params.data = torch.zeros_like(self.params.data)
#         elif init_params == "ones":
#             self.params.data = torch.ones_like(self.params.data)
#         elif init_params == "randn":
#             self.params.data = torch.randn_like(self.params.data)
#         elif init_params == "rand":
#             self.params.data = torch.rand_like(self.params.data)
#         elif type(init_params) in [int, float]:
#             self.params.data = torch.ones_like(self.params.data) * init_params
#         elif hasattr(init_params, "__call__"):
#             self.params.data = init_params(self.params.data)
#         else:
#             raise NotImplementedError
    
#     def all_named_parameters(self):
#         return [(k, v) for k, v in self.named_parameters()]

#     def forward(self, task=None):
#         if self.param_func is not None:
#             assert task is not None
#             return self.param_func(task=task, params=self.params)
#         else:
#             return self.params
