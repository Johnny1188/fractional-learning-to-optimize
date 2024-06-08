import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from fl2o.optimizee_modules import MetaModule, MetaLinear, MetaParameter, MetaBatchNorm1d

DEVICE = os.getenv("DEVICE", "cpu")


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
        batch_norm=False,
        output_bias=True,
        device=DEVICE,
    ):
        assert not batch_norm, "batch_norm not yet fully implemented"
        super().__init__()

        self.layer_sizes = layer_sizes
        self.inp_size = inp_size
        self.out_size = out_size
        self.act_fn = act_fn
        self.out_act_fn = out_act_fn
        self.batch_norm = batch_norm
        self.output_bias = output_bias

        ### init layers
        layers = []
        for i, layer_size in enumerate(layer_sizes):
            layers.append(MetaLinear(inp_size, layer_size, device=device))
            if batch_norm:
                layers.append(MetaBatchNorm1d(num_features=layer_size, device=device))
            layers.append(act_fn)
            inp_size = layer_size
        layers.append(MetaLinear(inp_size, out_size, bias=output_bias, device=device))
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
            batch_norm=self.batch_norm,
            output_bias=self.output_bias,
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

    def forward_w_params(self, params, params_for, start_at_layer_idx=None, stop_at_layer_idx=None, task=None, inp=None):
        ### allows to call forward() with a different set of params
        ### - first dim of params should be the number of params to run with
        assert task is not None or inp is not None

        ### prepare params to run with
        run_params = dict()
        if params_for is None:
            assert type(params) == dict
            run_params = params
        elif type(params_for) == str:
            run_params[params_for] = params
        else:
            for p_target, p in zip(params_for, ps):
                run_params[p_target] = p

        ### run
        x = inp.view(-1, self.inp_size) if inp is not None else task["x"].view(-1, self.inp_size)
        for l_idx, l in enumerate(self.layers):
            ### skip
            if start_at_layer_idx is not None and l_idx < start_at_layer_idx:
                continue

            ### stop earlier
            if stop_at_layer_idx is not None and l_idx >= stop_at_layer_idx:
                break

            ### run through the layer
            if "linear" in l.__class__.__name__.lower():
                if f"layers.{l_idx}.weight" in run_params:
                    ps = run_params[f"layers.{l_idx}.weight"]
                    if ps.ndim == 2:
                        x = x @ ps.T
                    elif ps.ndim == 3:
                        x = (ps @ x.T).permute(2, 0, 1)
                else:
                    x = x @ l.weight.T

                if f"layers.{l_idx}.bias" in run_params:
                    ps = run_params[f"layers.{l_idx}.bias"]
                    if ps.ndim == 1:
                        x += ps
                    else:
                        x = x.unsqueeze(1).expand(-1, ps.shape[0], -1) + ps
                elif l.bias is not None:
                    x += l.bias
            elif "batchnorm" in l.__class__.__name__.lower():
                if l.training:
                    m, v_unb, v_b = x.mean(0), x.var(0, unbiased=False), x.var(0, unbiased=True)
                    x = (x - m) / (torch.sqrt(v_unb) + l.eps) * l.weight
                    if l.bias is not None:
                        x += l.bias
                    # if l.running_mean.ndim != m.ndim:
                    #     l.running_mean = l.running_mean.unsqueeze(0).repeat(m.shape[0], 1)
                    # if l.running_var.ndim != v_b.ndim:
                    #     l.running_var = l.running_var.unsqueeze(0).repeat(v_b.shape[0], 1)
                    # l.running_mean = (1 - l.momentum) * l.running_mean + l.momentum * m
                    # l.running_var = (1 - l.momentum) * l.running_var + l.momentum * v_b
                else:
                    x = (x - l.running_mean) / (torch.sqrt(l.running_var) + l.eps) * l.weight
                    if l.bias is not None:
                        x += l.bias
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
