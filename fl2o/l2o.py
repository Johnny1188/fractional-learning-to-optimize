import os
from copy import deepcopy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

DEVICE = os.getenv("DEVICE", "cpu")


class L2O(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_sz,
        in_features,
        base_opter_cls,
        base_opter_config,
        params_to_optimize=None, # a dictionary of base opter attributes to optimize
    ):
        assert base_opter_cls.__name__ in ["FGD", "CFGD_ClosedForm", "CFGD", "L2O_Update"], \
            "base_opter_cls not supported"
        assert params_to_optimize is not None, "params_to_optimize must be specified"
        assert type(in_features) in [list, tuple], "in_features must be a list or tuple"
        assert len(in_features) <= in_dim, "in_features must have length equal or smaller than out_dim"
        assert type(params_to_optimize) == dict, "params_to_optimize must be a dictionary"
        super().__init__()

        self.in_dim = in_dim
        self.hidden_sz = hidden_sz
        self.out_dim = out_dim
        self.in_features = in_features
        self.base_opter_cls = base_opter_cls
        self.base_opter_config = base_opter_config
        self.params_to_optimize = params_to_optimize

        self.rnn1 = nn.LSTMCell(in_dim, hidden_sz)
        self.rnn2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.out_layer = nn.Linear(hidden_sz, out_dim)
        self.n_iters = None
        # self.l2o_inp_batch_dim = 1 if "grad" in self.in_features else None
        self.l2o_inp_batch_dim = 1
        self.per_param = "grad" in self.in_features  # whether to predict baseopter params for each param separately
        
        ### preproc params
        self.a, self.b, self.c = 2, 1, 3
        self.preproc_factor = 10.
        self.preproc_threshold = np.exp(-self.preproc_factor)

        ### init params
        torch.nn.init.xavier_uniform_(self.out_layer.weight)
        torch.nn.init.zeros_(self.out_layer.bias)

        ### init hidden and cell states
        self._reset_hidden()
        self._reset_cell()

        self.to(DEVICE)

    def _reset_hidden(self):
        self.hidden = [
            torch.zeros(self.l2o_inp_batch_dim, self.hidden_sz, device=DEVICE),
            torch.zeros(self.l2o_inp_batch_dim, self.hidden_sz, device=DEVICE)
        ]

    def _reset_cell(self):
        self.cell = [
            torch.zeros(self.l2o_inp_batch_dim, self.hidden_sz, device=DEVICE),
            torch.zeros(self.l2o_inp_batch_dim, self.hidden_sz, device=DEVICE)
        ]

    def _get_iter_num_enc(self, iter_num):
        ### return a value between 0 and 1 indicating how far along the optimization process we are
        assert self.n_iters is not None, "self.n_iters must be set before calling _get_iter_num_enc"
        return iter_num / self.n_iters

    def _preproc_grad(self, grad):
        # Implement preproc described in Appendix A from Andrychowicz et al. (2016)
        grad = grad.data
        grad_preproc = torch.zeros(grad.size()[0], 2, device=DEVICE)
        keep_grad_mask = (torch.abs(grad) >= self.preproc_threshold).squeeze(dim=-1)
        grad_preproc[:, 0][keep_grad_mask] = (
            torch.log(torch.abs(grad[keep_grad_mask]) + 1e-8) / self.preproc_factor
        ).squeeze()
        grad_preproc[:, 1][keep_grad_mask] = torch.sign(grad[keep_grad_mask]).squeeze()

        grad_preproc[:, 0][~keep_grad_mask] = -1
        grad_preproc[:, 1][~keep_grad_mask] = (
            float(np.exp(self.preproc_factor)) * grad[~keep_grad_mask]
        ).squeeze()

        return grad_preproc

    def _get_inp(self, y_hat, loss, task, optee, iter_num):
        ### get input for L2O
        inp = []
        resize_fn = lambda x: x.unsqueeze(0)  # resizing for features

        if "grad" in self.in_features:
            optee_grads = torch.cat([p.grad.view(-1, 1) for p in optee.parameters()], dim=0)
            optee_grads = self._preproc_grad(optee_grads)
            inp.append(optee_grads)
            resize_fn = lambda x: x.unsqueeze(0).expand(optee_grads.shape[0], -1)
        
        if "loss" in self.in_features:
            inp.append(resize_fn(loss))
        
        if "log_loss" in self.in_features:
            inp.append(resize_fn(loss.log()))

        if "iter_num_enc" in self.in_features:
            inp.append(resize_fn(torch.tensor(self._get_iter_num_enc(iter_num), dtype=torch.float32, device=DEVICE)))

        if "grad_mean_norm" in self.in_features:
            grads_mean = np.mean([p.grad.norm().item() for p in optee.parameters()])
            inp.append(resize_fn(torch.tensor(grads_mean, dtype=torch.float32, device=DEVICE)))

        if "grad_mean_abs" in self.in_features:
            grads_abs = np.mean([p.grad.abs().mean().item() for p in optee.parameters()])
            inp.append(resize_fn(torch.tensor(grads_abs, dtype=torch.float32, device=DEVICE)))

        if "log_grad_mean_abs" in self.in_features:
            log_grads_abs = np.mean([p.grad.abs().mean().log().item() for p in optee.parameters()])
            inp.append(resize_fn(torch.tensor(log_grads_abs, dtype=torch.float32, device=DEVICE)))

        if "log_grad_std" in self.in_features:
            log_grads_std = np.var([p.grad.std().log().item() for p in optee.parameters()])
            inp.append(resize_fn(torch.tensor(log_grads_std, dtype=torch.float32, device=DEVICE)))

        inp = torch.cat(inp, dim=-1).detach() # (B, inp_dim)
        if inp.ndim == 1:
            inp = inp.unsqueeze(0)
        return inp

    def zero_grad(self):
        raise NotImplementedError

    def prep_for_optim_run(self, optee, n_iters):
        self.base_opter = self.base_opter_cls(params=optee.parameters(), **self.base_opter_config)
        self.n_iters = n_iters
        if "grad" in self.in_features: # predicting baseopter params for each param separately
            self.l2o_inp_batch_dim = sum([p.numel() for p in optee.parameters()])
        self._reset_hidden()
        self._reset_cell()

    def finish_unroll(self):
        self.hidden = [h.detach() for h in self.hidden]
        self.cell = [c.detach() for c in self.cell]

    def out_act_fn(self, params_hat, config):
        assert "idx" in config, "config must have key 'idx'"
        assert "act_fns" in config, "config must have key 'act_fn'"

        out = params_hat[..., config["idx"]]
        for act_fn in config["act_fns"]:
            if act_fn == "sigmoid":
                out = torch.sigmoid(out)
            elif act_fn == "relu":
                out = F.relu(out)
            elif act_fn == "tanh":
                out = torch.tanh(out)
            elif act_fn == "abc_sigmoid":
                out = self.a / (self.b + torch.exp(-self.c * out))
            elif act_fn == "identity":
                pass
            elif act_fn == "diag":
                out = torch.diag(out)
            elif act_fn == "alpha_to_gamma":  # gamma = beta - (1 - alpha)/(2 - alpha)
                assert "beta" in config, "config must have key 'beta'"
                out = config["beta"] - (1 - out) / (2 - out)
            elif act_fn == "alpha_beta_to_gamma":  # gamma = beta - (1 - alpha)/(2 - alpha)
                assert out.shape[-1] == 2, "out.shape[-1] must be 2 for alpha_beta_to_gamma"
                alpha, beta = out[..., 0], out[..., 1]
                out = beta - (1 - alpha) / (2 - alpha)
            else:
                raise NotImplementedError
            
        return out

    def forward(self, x):
        h0, c0 = self.rnn1(x, (self.hidden[0], self.cell[0]))
        h1, c1 = self.rnn2(h0, (self.hidden[1], self.cell[1]))

        ### update hidden and cell states
        self.hidden = [h0, h1]
        self.cell = [c0, c1]

        return self.out_layer(h1)

    def step(self, y_hat, loss, task, optee, iter_num):
        ### get input for L2O
        inp = self._get_inp(y_hat=y_hat, loss=loss, task=task, optee=optee, iter_num=iter_num)
        params_hat = self(inp) # (B, out_dim)

        ### reshape to match optee's params
        if self.per_param:
            start_idx, end_idx = 0, 0
            params_hat_reshaped = []
            for g in self.base_opter.param_groups:
                end_idx += g["param_shape"].numel()
                params_hat_reshaped.append(params_hat[start_idx:end_idx, :].view(*g["param_shape"], -1))
                start_idx = end_idx

        ### step with base opter using params_hat
        for k, p_config in self.params_to_optimize.items():
            for g_idx, g in enumerate(self.base_opter.param_groups):
                if self.per_param:
                    p = params_hat_reshaped[g_idx]
                else:
                    p = params_hat
                g[k] = self.out_act_fn(params_hat=p, config=p_config)

                ### just for logging
                if k == "gamma" and "alpha_beta_to_gamma" in p_config["act_fns"]:
                    g["alpha"] = p[..., 0].detach().clone().cpu().numpy()
                    g["beta"] = p[..., 1].detach().clone().cpu().numpy()
                elif k == "gamma" and "alpha_to_gamma" in p_config["act_fns"]:
                    g["alpha"] = p.detach().clone().cpu().numpy()
                    g["beta"] = p_config["beta"] * np.ones_like(p.detach().clone().cpu().numpy())

        return self.base_opter.step(task=task, optee=optee)
