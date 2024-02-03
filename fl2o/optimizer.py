from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from scipy.special import gamma
from scipy.special import roots_jacobi
import numpy as np

from fl2o.utils import rsetattr, rgetattr, roots_jacobi_vectorized


class FGD(nn.Module):
    """
    Fractional (Stochastic) Gradient Descent with Higher Order Truncation (hot) and/or Fixed Memory Step (K>=1)
    - ref: https://arxiv.org/abs/1901.05294v2 (algorithms 1 and 2)
    """

    def __init__(
        self,
        params,
        lr=0.01,
        alpha=1,
        c=0,
        eps=1e-8,
        hot=True,
        K=0,
        device="cpu",
    ):
        assert hot, "Only hot=True is supported for now."
        super().__init__()

        self.param_groups = [
            {"params": None, "lr": lr, "alpha": alpha, "c": c, "eps": eps, "hot": hot, "K": K, "device": device}
        ]
        self.state = []
        for p in params:
            p.retain_grad()
            self.state.append({})
        self.hot = hot
        self.K = K
        self.device = device

    def zero_grad(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    # p.grad.detach_()
                    p.grad.zero_()
                else:
                    print(f"[WARNING] p.grad is None for {p}.")

    def step(self, optee):
        for g in self.param_groups:
            ### update params
            updated_params = {}
            for p_idx, (n, p) in enumerate(optee.all_named_parameters()):
                if p.grad is None:
                    continue
                if len(self.state[p_idx]) == 0:
                    self.state[p_idx]["step"] = 1 # iter starts from 1
                    self.state[p_idx]["memory"] = [g["c"] for _ in range(max(1, g["K"]))]

                ### get update 'h'
                grad = p.grad.data
                tmp = ((p.data.detach() - self.state[p_idx]["memory"][0]).abs() + g["eps"])**(1 - g["alpha"])
                if type(g["alpha"]) in (float, int):
                    gam = gamma(2 - g["alpha"])
                elif type(g["alpha"]) == torch.Tensor:
                    gam = torch.lgamma(2 - g["alpha"]).exp()
                else:
                    raise Exception("This should not happen.")
                h = grad * tmp / gam

                ### update params
                # p.add_(h, alpha=-g["lr"])
                updated_params[n] = p - g["lr"] * h
                updated_params[n].retain_grad()

                ### update integ_term according to the fixed memory step (K)
                if g["K"] >= 1:
                    self.state[p_idx]["memory"].pop(0)
                    self.state[p_idx]["memory"].append(p.data.detach().clone())
                self.state[p_idx]["step"] += 1
            
            ### update params
            if hasattr(optee, "layers"):
                for l_idx in range(len(optee.layers)):
                    if len(list(optee.layers[l_idx].parameters())) == 0:
                        continue
                    optee.layers[l_idx].weight = updated_params[f"layers.{l_idx}.weight"]
                    optee.layers[l_idx].bias = updated_params[f"layers.{l_idx}.bias"]
            else:
                for n, _ in optee.named_parameters():
                    setattr(optee, n, updated_params[n])
                    # p.data = updated_params[n]

        return None


class AFOGD(optim.Optimizer):
    """
    Adaptive Fractional Order (Accelerated) Gradient Descent
    - ref: https://arxiv.org/pdf/2303.04328v1.pdf
    """
    def __init__(
        self,
        params,
        lr=0.01,
        alpha=1,
        c1=0.7,
        c2=1.3,
        eps=1e-5,
        accelerated=False,
        ni=0.9,
        device="cpu",
    ):
        assert c1 < c2, "c1 must be less than c2."
        defaults = dict(
            lr=lr,
            alpha=alpha,
            c1=c1,
            c2=c2,
            eps=eps,
            accelerated=accelerated,
            device=device,
        )
        super().__init__(params, defaults)
        self.c1 = c1
        self.c2 = c2
        self.accelerated = accelerated
        self.ni = ni
        self.device = device

    def __setstate__(self, state):
        super().__setstate__(state)

    def _find_beta(self, alpha_dist_to_last_step, min_gap=1e-3):
        ### find beta such that: c1 <= beta * alpha_dist_to_last_step <= c2
        beta = (self.c1 + self.c2) / 2
        while (self.c1 > beta * alpha_dist_to_last_step) \
           or (self.c2 < beta * alpha_dist_to_last_step):
            if self.c1 > beta * alpha_dist_to_last_step:
                beta = self.c1 / alpha_dist_to_last_step + min_gap
            elif self.c2 < beta * alpha_dist_to_last_step:
                beta = self.c2 / alpha_dist_to_last_step - min_gap
            else:
                raise Exception("This should not happen.")
        return beta

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for g in self.param_groups:
            ### get params
            params = g["params"]

            ### update params
            for p in params:
                if p.grad is None:
                    continue
                if len(self.state[p]) == 0:
                    self.state[p]["step"] = 1 # iter starts from 1
                    self.state[p]["last_step"] = 0
                    self.state[p]["last_y"] = 0 if self.accelerated else None

                ### get update 'd'
                if self.accelerated:
                    y = p.data + self.ni * (p.data - self.state[p]["last_step"])
                    dist_to_last_step = torch.norm(y - self.state[p]["last_y"], p=2)
                    grad = p.grad.data # TODO: eval with y
                else:
                    dist_to_last_step = torch.norm(p.data - self.state[p]["last_step"], p=2)
                    grad = p.grad.data

                d = grad
                if self.state[p]["step"] > 1:
                    grad_mul = (dist_to_last_step + g["eps"])**(1 - g["alpha"]) + g["eps"]
                    grad_mul *= self._find_beta(alpha_dist_to_last_step=grad_mul) # find current beta
                    d *= grad_mul

                ### update params
                if self.accelerated:
                    p.data.fill_(y - g["lr"] * d)
                else:
                    p.data.add_(d.detach(), alpha=-g["lr"])

                ### update integ_term according to the fixed memory step (K)
                self.state[p]["last_step"] = p.data.detach().clone()
                if self.accelerated:
                    self.state[p]["last_y"] = y.detach().clone()
                self.state[p]["step"] += 1

        return loss


class CFGD(nn.Module):
    """
    Caputo Fractional Gradient Descent (CFGD)
    - ref: https://arxiv.org/abs/2104.02259
    """
    def __init__(
        self,
        params,
        lr=0.01,
        alpha=1, # fractional order
        beta=0,
        c=0, # integral terminal
        s=3, # number of sample points
        version="NA", # NA: non-adaptive, AT: adaptive-terminal
        init_points=None, # initial points for AT version
        max_chunk_size=5_000_000, # max number of params to fit into memory
        device="cpu",
    ):
        assert version.upper() in ("NA", "AT"), "Only NA and AT versions of CFGD are supported for now."
        assert version.upper() != "AT" or init_points is not None, "init_points must be provided for AT version."

        super().__init__()

        self.param_groups = []
        self.state = []
        p_idx = 0
        for p in params:
            p.retain_grad()
            self.state.append({})
            self.param_groups.append({
                "param_shape": p.shape,
                "lr": lr,
                "alpha": alpha,
                "beta": beta,
                "c": c,
                "s": s,
                "device": device,
                "version": version.upper(),
                "init_points": init_points[p_idx] if init_points is not None else None,
            })
            p_idx += 1

        self.s = s
        self.max_chunk_size = max_chunk_size
        self.version = version.upper()
        self.device = device

    def _get_grads_diag_hess(self, optee, task, forward_w_params, params_for, n_iters=3):
        ### set requires_grad=False for all params
        required_grad = dict()
        for n, p in optee.all_named_parameters():
            required_grad[n] = p.requires_grad
            p.requires_grad = False

        ### get grads
        y_hat = optee.forward_w_params(task=task, params=forward_w_params, params_for=params_for)
        if "A" in task.keys():  # quadratic objective function -> add data batch dim
            y_hat = y_hat.unsqueeze(0)
            loss = sum([task["loss_fn"](y_hat=y_hat[:,p_idx,:]) for p_idx in range(y_hat.shape[1])])
        else:
            loss_fn = task["loss_fn_cls"](reduction="sum")
            loss = loss_fn(
                y_hat.permute(0, 2, 1), # (data_batch_size, n_classes, n_params_to_try)
                task["y"].unsqueeze(-1).expand(-1, y_hat.shape[1]), # (data_batch_size, n_classes)
            ) / y_hat.shape[0] # average over data batch size
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=forward_w_params,
            create_graph=True,
        )[0]

        ### approximate diag(H)
        Hd = torch.zeros_like(forward_w_params, requires_grad=False)  # (n_params_to_try, param_shape*)
        for iter_i in range(n_iters):
            ### v ~ rademacher distribution
            v = torch.rand_like(forward_w_params, requires_grad=False)  # (n_params_to_try, param_shape*)
            v[v < 0.5] = -1
            v[v >= 0.5] = 1

            ### get the H @ v
            vprod = (v * grads).sum()
            Hv = torch.autograd.grad(
                outputs=vprod,
                inputs=forward_w_params,
                retain_graph=iter_i < n_iters - 1
            )[0]

            ### update Hd
            Hd += (Hv * v).abs()
        Hd /= n_iters

        ### set requires_grad back to the original values
        for n, p in optee.all_named_parameters():
            p.requires_grad = required_grad[n]

        return grads, Hd

    @staticmethod
    def get_c_alpha_beta(alpha, beta):
        return (1 - alpha) / (
            (1 + abs(beta)) * (2**(1 - alpha))
        )

    def zero_grad(self):
        raise NotImplementedError("zero_grad() is not yet supported.")

    def step(self, task, optee):
        optee_to_use = optee.get_deepcopy()

        ### Gauss-Jacobi quadrature
        # sample_points, sample_weights = roots_jacobi(n=self.s, alpha=-0.85, beta=0, mu=False)
        # sample_weights = torch.tensor(sample_weights, device=self.device).detach().float()
        # sample_points = torch.tensor(sample_points, device=self.device).detach().float()

        ### find out where to eval fo and so partial derivatives
        deriv_eval_points = [] # fill to (n_params, param_size, s)
        sample_points, sample_weights = dict(), dict()
        for p_idx, (n, p) in enumerate(optee_to_use.all_named_parameters()):
            ### prepare internal state
            if len(self.state[p_idx]) == 0 or "step" not in self.state[p_idx]:
                self.state[p_idx]["step"] = 1 # iter starts from 1
                if self.version == "AT":
                    self.state[p_idx]["memory"] = [init_p.detach().clone() for init_p in self.param_groups[p_idx]["init_points"]]
            else:
                self.state[p_idx]["step"] += 1
            
            ### get sample points and weights for the Gauss-Jacobi quadrature # TODO: vectorize
            if type(self.param_groups[p_idx]["alpha"]) in (float, int):
                sample_points[n], sample_weights[n] = roots_jacobi(n=self.s, alpha=-self.param_groups[p_idx]["alpha"], beta=0, mu=False)
                sample_weights[n] = torch.tensor(sample_weights[n], device=self.device).detach().float()
                sample_points[n] = torch.tensor(sample_points[n], device=self.device).detach().float()
            elif type(self.param_groups[p_idx]["alpha"]) == torch.Tensor:
                ##### DEPRECATED (too slow) #####
                # sample_points[n], sample_weights[n] = [], []
                # for alpha in self.param_groups[p_idx]["alpha"].view(-1):
                #     roots = roots_jacobi(n=self.s, alpha=-alpha.item(), beta=0, mu=False)
                #     sample_points[n].append(roots[0])
                #     sample_weights[n].append(roots[1])
                # ### reshape to (param_size, s)
                # sample_weights[n] = torch.tensor(sample_weights[n], device=self.device).detach().float().view(*p.shape, self.s)
                # sample_points[n] = torch.tensor(sample_points[n], device=self.device).detach().float().view(*p.shape, self.s)
                #####
                sample_points[n], sample_weights[n] = roots_jacobi_vectorized(
                    N=self.s,
                    alpha=-self.param_groups[p_idx]["alpha"].detach().view(-1),
                    beta=torch.zeros_like(self.param_groups[p_idx]["alpha"].view(-1)),
                )
                ### reshape to (param_size, s)
                sample_points[n] = sample_points[n].view(*p.shape, self.s)
                sample_weights[n] = sample_weights[n].view(*p.shape, self.s)

            c = self.param_groups[p_idx]["c"] if self.version == "NA" else self.state[p_idx]["memory"][0]
            delta_plus = (p.data.detach() + c).mul(0.5).unsqueeze(-1) # (param_size, 1)
            delta_minus = (p.data.detach() - c).mul(0.5).unsqueeze(-1) # (param_size, 1)
            deriv_eval_points.append(
                delta_plus * sample_points[n] + delta_minus
            ) # (param_size, s)

        ### get fo and so partial derivatives at s sample points
        fos = [[] for _ in range(len(optee_to_use.all_named_parameters()))] # fill to (n_params, s, param_size, 1)
        sos = [[] for _ in range(len(optee_to_use.all_named_parameters()))]
        compute_so = [True for _ in range(len(optee_to_use.all_named_parameters()))]
        for s_idx in range(self.s):
            ### get fo and so partial derivatives to all elems in each param tensor (separately)
            for p_idx, (n, p) in enumerate(optee_to_use.all_named_parameters()):
                if (type(self.param_groups[p_idx]["beta"]) in (float, int) and self.param_groups[p_idx]["beta"] == 0) \
                    or (type(self.param_groups[p_idx]["beta"]) == torch.Tensor and self.param_groups[p_idx]["beta"].max().item() == 0):
                    compute_so[p_idx] = False

                ### change individual params while keeping all the other params fixed -> parameter batch dim B == param_size
                ### split the number of params (the batch size B) into chunks of size 'chunk_size' to fit into memory
                n_elems = p.shape.numel()
                chunks_fos, chunks_sos = [], []
                for start_idx in range(0, n_elems, int(np.sqrt(self.max_chunk_size))):
                    end_idx = min(start_idx + int(np.sqrt(self.max_chunk_size)), n_elems) + 1
                    B = end_idx - start_idx - 1  # current batch dim

                    # batch of params
                    pp = p.data.detach().clone().view(-1).unsqueeze(0).repeat(B, 1)
                    changed_elems = torch.arange(start_idx, end_idx - 1)
                    # change the params
                    pp[torch.arange(B), changed_elems] = deriv_eval_points[p_idx][..., s_idx].view(-1)[start_idx:start_idx + B]
                    pp.requires_grad_(True)
                    # get fo and so partial derivatives
                    curr_fos, curr_sos = self._get_grads_diag_hess(
                        optee=optee_to_use,
                        task=task,
                        forward_w_params=pp.view(B, *p.shape),  # resize to (B, param_shape*) for the forward pass
                        params_for=n,
                    )

                    # extract only the fo and so derivatives wrt. to the changed params
                    chunks_fos.append(
                        curr_fos.detach().view(B, p.shape.numel())[torch.arange(B), changed_elems]
                    )
                    chunks_sos.append(
                        curr_sos.detach().view(B, p.shape.numel())[torch.arange(B), changed_elems]
                    )
                fos[p_idx].append(torch.cat(chunks_fos).view(*p.shape))
                if compute_so[p_idx]:
                    sos[p_idx].append(torch.cat(chunks_sos).view(*p.shape))

        ### get updated params
        updated_params = dict()
        for p_idx, (n, p) in enumerate(optee.all_named_parameters()):
            ### prepare update 'd'
            g = self.param_groups[p_idx]
            to_sum = torch.stack(fos[p_idx]) # (s, param_size)

            if compute_so[p_idx]:
                c = g["c"] if self.version == "NA" else self.state[p_idx]["memory"][0]
                to_sum += g["beta"] * (p.data - c).abs() * torch.stack(sos[p_idx]) # (s, param_size)

            C_alpha_beta = self.get_c_alpha_beta(alpha=g["alpha"], beta=g["beta"])
            d = C_alpha_beta * to_sum.movedim(0, -1).mul(sample_weights[n]).sum(dim=-1) # (param_size)

            ### get updated params
            lr = g["lr"]
            if hasattr(lr, "__call__"):
                lr = lr( # only for quadratic objective functions
                    A=task["A"],
                    b=task["b"],
                    x=p.detach().data.T,
                    d=d.detach().T,
                )
            updated_params[n] = p - lr * d
            updated_params[n].retain_grad()

            ### update internal state
            if self.version == "AT":
                self.state[p_idx]["memory"].pop(0)
                self.state[p_idx]["memory"].append(p.data.detach().clone())
            self.state[p_idx]["last_update"] = lr * d.detach()
            self.state[p_idx]["last_grad"] = p.grad.detach().clone() if p.grad is not None else None
            self.state[p_idx]["last_lr"] = lr

        ### update params
        for n, _ in optee.all_named_parameters():
            rsetattr(optee, n, updated_params[n])

        return None


class CFGD_ClosedForm(nn.Module):
    """
    Caputo Fractional Gradient Descent (CFGD) - Closed Form (only for quadratic functions)
    - ref: https://arxiv.org/abs/2104.02259

    - note: Now only non-adaptive and adaptive-terminal versions are supported.
    """

    def __init__(
        self,
        params,
        lr=0.1,
        gamma=0, # fractional order alpha and beta combined: gamma = beta - (1 - alpha)/(2 - alpha)
        c=1, # integral terminal
        version="NA", # NA: non-adaptive, AT: adaptive-terminal
        init_points=None, # initial points for AT version
        device="cpu",
    ):
        assert version.upper() in ("NA", "AT"), "Only NA and AT versions of CFGD are supported for now."
        assert version.upper() != "AT" or init_points is not None, "init_points must be provided for AT version."
        super().__init__()

        self.param_groups = []
        self.state = []
        for p in params:
            p.retain_grad()
            self.state.append({})
            self.param_groups.append(
                {
                    "param_shape": p.shape,
                    "lr": lr,
                    "gamma": gamma,
                    "c": c,
                    "device": device,
                    "version": version.upper(),
                    "init_points": init_points,
                }
            )

        self.version = version.upper()
        self.device = device

    def zero_grad(self):
        raise NotImplementedError("zero_grad() is not supported for CFGD_ClosedForm.")
        # for p_idx, (n, p) in enumerate(optee.all_named_parameters()):
        #     if p.grad is not None:
        #         p.grad.zero_()
        #     else:
        #         print(f"[WARNING] p.grad is None for {p}.")

        # for g in self.param_groups:
        #     for p in g["params"]:
        #         if p.grad is not None:
        #             # p.grad.detach_()
        #             p.grad.zero_()
        #         else:
        #             print(f"[WARNING] p.grad is None for {p}.")

    def get_R_tilde(self, task):
        R_tilde = torch.diag(torch.sqrt(torch.diag(task["A"])))
        return R_tilde

    def step(self, task, optee, closure=None):
        """
        Closed form step for quadratic functions from Corollary 1
        """
        loss = None
        if closure is not None:
            loss = closure()

        ### get params for the closed form update
        A, b = task["A"], task["b"]
        R_tilde = self.get_R_tilde(task).to(self.device)

        ### get updated params
        updated_params = {}
        for p_idx, (g, (n, p)) in enumerate(zip(self.param_groups, optee.all_named_parameters())):
            if p.grad is None:
                continue

            if len(self.state[p_idx]) == 0:
                self.state[p_idx]["step"] = 1 # iter starts from 1
                if self.version == "AT":
                    self.state[p_idx]["memory"] = [init_p.detach().clone() for init_p in g["init_points"]]

            ### get update 'd'
            c = g["c"] if self.version == "NA" else self.state[p_idx]["memory"][0]
            d = A @ p.data.T + b
            d += g["gamma"] * R_tilde @ (p.data - c).T
            d = d.T

            ### update params
            lr = g["lr"]
            if hasattr(lr, "__call__"):
                lr = lr(
                    A=A,
                    b=b,
                    x=p.detach().data.T,
                    d=d.detach().T,
                )
            updated_params[n] = p - lr * d
            updated_params[n].retain_grad()
            
            ### update internal state
            if self.version == "AT":
                self.state[p_idx]["memory"].pop(0)
                self.state[p_idx]["memory"].append(p.data.detach().clone())
            self.state[p_idx]["last_update"] = lr * d.detach()
            self.state[p_idx]["last_grad"] = p.grad.detach().clone()
            self.state[p_idx]["last_lr"] = lr
            self.state[p_idx]["step"] += 1

        ### update params
        for n, _ in optee.all_named_parameters():
            rsetattr(optee, n, updated_params[n])

        return loss


class GD(optim.Optimizer):
    """
    Gradient Descent (GD)
    """
    def __init__(
        self,
        params,
        lr=0.1,
        device="cpu",
    ):
        defaults = dict(
            lr=lr,
            device=device,
        )
        super().__init__(params, defaults)

        self.device = device

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, task, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for g in self.param_groups:
            ### get params
            params = g["params"]

            ### update params
            for p in params:
                if p.grad is None:
                    continue

                ### get update 'd'
                d = p.grad.data

                ### update params
                lr = g["lr"]
                if hasattr(lr, "__call__"):
                    assert task is not None, "task must be provided for lr function."
                    assert "A" in task and "b" in task, "task must have A and b."
                    A, b = task["A"], task["b"]
                    lr = lr(
                        A=A,
                        b=b,
                        x=p.data.T,
                        d=d.T,
                    )
                p.data.add_(d.detach(), alpha=-lr)
                g["last_update"] = lr * d.detach()
                g["last_grad"] = p.grad.detach().clone()
                g["last_lr"] = lr

        return loss


class Adam(optim.Optimizer):
    """
    Adam
    """
    def __init__(
        self,
        params,
        lr=0.1,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        device="cpu",
    ):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            device=device,
        )
        super().__init__(params, defaults)

        self.device = device

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, task, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for g in self.param_groups:
            ### get params
            params = g["params"]

            ### init tracking
            if "step" not in g:
                g["step"] = 0
            if "exp_avg" not in g:
                g["exp_avg"] = torch.zeros_like(params[0].data)
            if "exp_avg_sq" not in g:
                g["exp_avg_sq"] = torch.zeros_like(params[0].data)

            ### update params
            for p in params:
                if p.grad is None:
                    continue

                ### get update 'd'
                d = p.grad.data

                ### update params
                lr = g["lr"]
                if hasattr(lr, "__call__"):
                    assert task is not None, "task must be provided for lr function."
                    assert "A" in task and "b" in task, "task must have A and b."
                    A, b = task["A"], task["b"]
                    lr = lr(
                        A=A,
                        b=b,
                        x=p.data.T,
                        d=d.T,
                    )
                g["step"] += 1
                g["exp_avg"] = g["beta1"] * g["exp_avg"] + (1 - g["beta1"]) * d
                g["exp_avg_sq"] = g["beta2"] * g["exp_avg_sq"] + (1 - g["beta2"]) * d**2
                exp_avg_hat = g["exp_avg"] / (1 - g["beta1"]**g["step"])
                exp_avg_sq_hat = g["exp_avg_sq"] / (1 - g["beta2"]**g["step"])
                p.data.add_(exp_avg_hat / (exp_avg_sq_hat.sqrt() + g["eps"]), alpha=-lr)

                g["last_update"] = lr * exp_avg_hat / (exp_avg_sq_hat.sqrt() + g["eps"])
                g["last_grad"] = p.grad.detach().clone()
                g["last_lr"] = lr

        return loss


class L2O_Update(nn.Module):
    """
    Placeholder Optimizer for applying L2O updates.
    """

    def __init__(
        self,
        params,
        lr=0.1,
        device="cpu",
    ):
        super().__init__()

        self.param_groups = []
        self.state = []
        for p in params:
            p.retain_grad()
            self.state.append({})
            self.param_groups.append(
                {
                    "param_shape": p.shape,
                    "lr": lr,
                    "device": device,
                }
            )

        self.device = device
    
    def zero_grad(self):
        raise NotImplementedError("zero_grad() is not supported for CFGD_ClosedForm.")

    def step(self, task, optee):
        ### get updated params
        updated_params = {}
        for p_idx, (g, (n, p)) in enumerate(zip(self.param_groups, optee.all_named_parameters())):
            if len(self.state[p_idx]) == 0:
                self.state[p_idx]["step"] = 1 # iter starts from 1
            else:
                self.state[p_idx]["step"] += 1

            ### get update 'd'
            d = g["update"]  # expected to be a tensor of the same shape as p

            ### get updated params
            lr = g["lr"]
            if hasattr(lr, "__call__"):
                lr = lr( # only for quadratic objective functions
                    A=task["A"],
                    b=task["b"],
                    x=p.detach().data.T,
                    d=d.detach().T,
                )
            updated_params[n] = p - lr * d
            updated_params[n].retain_grad()

            ### update internal state
            self.state[p_idx]["last_update"] = lr * d.detach()
            self.state[p_idx]["last_grad"] = p.grad.detach().clone() if p.grad is not None else None
            self.state[p_idx]["last_lr"] = lr

        ### update params
        for n, _ in optee.all_named_parameters():
            rsetattr(optee, n, updated_params[n])

        return None
