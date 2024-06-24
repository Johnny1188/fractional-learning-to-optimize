import copy
import json
import os
import time
import dill

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

DEVICE = os.getenv("DEVICE", "cpu")


def do_fit(
    opter_cls,
    opter_config,
    optee_cls,
    optee_config,
    data_cls,
    data_config,
    n_iters,
    l2o_dict=None,
    in_meta_training=False,
    additional_metrics=None,
    device=DEVICE,
):
    ### init optee
    optee = optee_cls(**optee_config)
    optee.train()

    ### init opter
    if l2o_dict is not None:
        opter = l2o_dict["opter"]
        opter.prep_for_optim_run(optee=optee, n_iters=n_iters)

        if in_meta_training:
            meta_opter = l2o_dict["meta_opter"]
            unroll = l2o_dict["unroll"]
            meta_opter.zero_grad()
            opter.train()
        else:
            if l2o_dict["unroll"] > 1:
                print(f"[WARNING] l2o_dict['unroll'] > 1 but in_meta_training is False. Setting l2o_dict['unroll'] to 1.")
            unroll = 1
            opter.eval()
    else:
        opter = opter_cls(params=optee.parameters(), **opter_config)
        unroll = 1

    ### init data
    data = data_cls(**data_config)

    ### init log
    log = {"loss": [], "time": []}
    if additional_metrics is not None:
        for m in additional_metrics:
            log[m] = []

    ### init time
    start_time = time.time()

    ### fit
    unroll_meta_loss = 0
    for iter_num in range(n_iters):
        ### zero grads
        # opter.zero_grad()
        for _, (n, p) in enumerate(optee.all_named_parameters()):
            if p.grad is not None:
                del p.grad

        ### get data
        task_sample = data.sample()

        ### get loss
        y_hat = optee(task=task_sample)
        loss = task_sample["loss_fn"](y_hat=y_hat)
        loss.backward(retain_graph=in_meta_training is True)

        ### check for NaNs in optee grads
        for p in optee.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f"[WARNING] NaN in p.grad. Skipping step.")
                return log, optee, opter

        ### check for NaN in opter
        if l2o_dict is not None and in_meta_training is True:
            for p in opter.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    print(f"[WARNING] NaN in p.grad. Skipping step.")
                    return log, optee, opter

        ### do step
        if opter.__class__.__name__.lower() in ("cfgd",):
            opter.step(task=task_sample, optee=optee)
        elif opter.__class__.__name__.lower() in ("fgd", "afogd",):
            opter.step()
        elif opter.__class__.__name__.lower() in ("l2o",):
            opter.step(y_hat=y_hat, loss=loss, task=task_sample, optee=optee, iter_num=iter_num)
            unroll_meta_loss = unroll_meta_loss + loss
        elif opter.__class__.__name__.lower() in ("cfgd_closedform",):
            opter.step(task=task_sample, optee=optee)
        elif opter.__class__.__name__.lower() in ("gd",):
            opter.step(task=task_sample, optee=optee)
        elif opter.__class__.__name__.lower() in ("adam",):
            opter.step(task=task_sample)
        else:
            opter.step()

        ### log
        log["loss"].append(loss.item())
        log["time"].append(time.time() - start_time)
        if additional_metrics is not None:
            for m in additional_metrics:
                log[m].append(additional_metrics[m](y_hat=y_hat, task=task_sample, optee=optee, iter_num=iter_num, opter=opter))

        ### end of unroll
        if (iter_num + 1) % unroll == 0 and l2o_dict is not None:
            ### update opter (l2o)
            if in_meta_training is True:
                meta_opter.zero_grad()
                torch.log(unroll_meta_loss).backward()
                meta_opter.step()
            unroll_meta_loss = 0

            opter.finish_unroll()

            ### update optee w/ detached params (important for l2o meta-training)
            if hasattr(optee, "layers"):
                for l_idx in range(len(optee.layers)):
                    for n, _ in optee.layers[l_idx].named_parameters():
                        setattr(
                            optee.layers[l_idx],
                            n,
                            getattr(optee.layers[l_idx], n).detach().requires_grad_(True),
                        )
                        getattr(optee.layers[l_idx], n).retain_grad()
            else:
                for n, _ in optee.all_named_parameters():
                    setattr(
                        optee,
                        n,
                        getattr(optee, n).detach().requires_grad_(True),
                    )
                    getattr(optee, n).retain_grad()
    
    return log, optee, opter


def meta_train(config, l2o_dict=None, l2o_dict_best=None, log=None, device=DEVICE):
    print(f"[INFO] Running on {device}")
    _config = copy.deepcopy(config)

    ### keep meta-training previous
    if l2o_dict is not None:
        assert l2o_dict_best is not None and log is not None
        print(f"[INFO] Continuing meta-training.")
    else: # first meta-training
        print(f"[INFO] Starting meta-training.")
        
        ### init opter and meta_opter
        opter = _config["opter"]["opter_cls"](**_config["opter"]["opter_config"])
        meta_opter = _config["meta_training_config"]["meta_opter_cls"](
            params=opter.parameters(),
            **_config["meta_training_config"]["meta_opter_config"]
        )

        ### init l2o_dict
        l2o_dict = {
            "opter": opter,
            "meta_opter": meta_opter,
            "unroll": _config["meta_training_config"]["unroll"],
        }
        l2o_dict_best = {"best_loss_sum": np.inf, "best_l2o_dict": None, "run_num": None} # early stopping

        ### init log
        log = {k: [] for k in 
            ["loss", "acc", "loss_sum", "loss_last", "acc_sum", "acc_last"]
        }

    ### meta-train
    n_runs = _config["meta_training_config"]["n_runs"]
    l2o_dict["opter"].train()
    print(f"[INFO] Meta-training starts.")
    for run_num in range(1, n_runs + 1):
        print(f"  [{run_num}/{n_runs}]> ", end="")
        _log, _, _ = do_fit(
            opter_cls=None,
            opter_config=None,
            optee_cls=_config["optee"]["optee_cls"],
            optee_config=_config["optee"]["optee_config"],
            data_cls=_config["data"]["data_cls"],
            data_config=_config["data"]["data_config"],
            n_iters=_config["n_iters"],
            l2o_dict=l2o_dict,
            in_meta_training=True,
            additional_metrics=_config["additional_metrics"],
            device=device,
        )
        log["loss"].append(_log["loss"])
        log["loss_sum"].append(np.sum(_log["loss"]))
        log["loss_last"].append(_log["loss"][-1])

        ### log metrics
        print(
            f"sum(loss): {log['loss_sum'][-1]:.3f}"
            f"  last(loss): {log['loss_last'][-1]:.3f}"
        )
        if _config["meta_training_config"]["loggers"] is not None:
            for logger in _config["meta_training_config"]["loggers"]:
                if run_num % logger["every_nth_run"] == 0:
                    logger["logger_fn"](run_log=_log, meta_training_log=log, run_num=run_num)

        ### early stopping
        if log["loss_sum"][-1] < l2o_dict_best["best_loss_sum"]:
            l2o_dict_best["best_loss_sum"] = log["loss_sum"][-1]
            l2o_dict_best["best_l2o_dict"] = {
                "opter_state_dict": copy.deepcopy(l2o_dict["opter"].state_dict()),
                "meta_opter_state_dict": copy.deepcopy(l2o_dict["meta_opter"].state_dict()),
                "unroll": l2o_dict["unroll"],
            }
            l2o_dict_best["run_num"] = run_num
            print(f"       > new best loss sum: {l2o_dict_best['best_loss_sum']:.3f}")

        ### save checkpoint
        ckpt_config = _config["ckpt_config"]
        if ckpt_config is not None:
            assert "ckpt_every_nth_run" in ckpt_config, "ckpt_every_nth_run not in ckpt_config"
            if ckpt_config["ckpt_every_nth_run"] is not None and run_num % ckpt_config["ckpt_every_nth_run"] == 0:
                print(f"       > saving checkpoint")
                torch.save({
                        "l2o_dict": l2o_dict,
                        "l2o_dict_best": l2o_dict_best,
                        "log": log,
                        "config": _config,
                    }, os.path.join(ckpt_config["ckpt_dir_meta_training"], f"{run_num}.pt"),
                    pickle_module=dill,
                )

    ### finalize l2o_dict_best
    l2o_dict_best["best_l2o_dict"]["opter"] = _config["opter"]["opter_cls"](**_config["opter"]["opter_config"])
    l2o_dict_best["best_l2o_dict"]["opter"].load_state_dict(l2o_dict_best["best_l2o_dict"]["opter_state_dict"])
    l2o_dict_best["best_l2o_dict"]["meta_opter"] = _config["meta_training_config"]["meta_opter_cls"](
        params=l2o_dict_best["best_l2o_dict"]["opter"].parameters(),
        **_config["meta_training_config"]["meta_opter_config"],
    )
    l2o_dict_best["best_l2o_dict"]["meta_opter"].load_state_dict(l2o_dict_best["best_l2o_dict"]["meta_opter_state_dict"])

    return l2o_dict, l2o_dict_best, log


def find_best_lr(
    opter_cls,
    opter_config,
    optee_cls,
    optee_config,
    data_cls,
    data_config,
    loss_fn,
    n_iters=50,
    n_tests=1,
    consider_metric="loss",
    lrs_to_try=None,
):
    assert consider_metric in ["loss"] # TODO: add test_loss/train_loss

    opter_config = copy.deepcopy(opter_config) if opter_config is not None else {}
    if lrs_to_try is None:
        lrs_to_try = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 7e-1, 1e0, 3e0]

    best_loss_sum = np.inf
    best_lr = None

    for lr in lrs_to_try:
        opter_config["lr"] = lr

        ### run n_tests
        test_loss_sums = []
        for _ in range(n_tests):
            log, _, _ = do_fit(
                opter_cls=opter_cls,
                opter_config=opter_config,
                optee_cls=optee_cls,
                optee_config=optee_config,
                data_cls=data_cls,
                data_config=data_config,
                loss_fn=loss_fn,
                n_iters=n_iters,
            )
            test_loss_sums.append(np.sum(log[consider_metric]))

        ### check if best
        loss_sum = np.mean(test_loss_sums)
        if loss_sum < best_loss_sum:
            best_loss_sum = loss_sum
            best_lr = lr

    return best_lr


def get_optimal_lr(
    A, # model matrix
    b, # model bias
    x, # current iterate
    d, # current update direction
):
    """Compute optimal step size for a quadratic model."""
    lr = ((A @ x + b).T @ d) / (d.T @ A @ d + 1e-6)
    if type(lr) == torch.Tensor:
        lr = lr.item()
    return lr


def n_step_lookahead_lr_search_hfunc_tanh_twolayer_optee(
    task,
    optee,
    g,
    n_steps=1,
    lrs_to_try=None,
):
    assert n_steps == 1, "More than one step lookahead not yet implemented."

    lr_out = dict()
    with torch.no_grad():
        ### quadratic wrt to `layers.2.weight`
        first_layer_out = optee.forward_w_params(
            params={
                "layers.0.weight": optee.layers[0].weight,
                "layers.0.bias": optee.layers[0].bias,
            },
            params_for=None,
            stop_at_layer_idx=2, # stop before applying this weight matrix
            task=task,
        ) # (B, hidden_dim=N)
        A = first_layer_out.T @ first_layer_out # (N, N)
        b = -1 * first_layer_out.T @ task["y"] # (N)
        lr_out["layers.2.weight"] = get_optimal_lr(A=A, b=b, x=optee.layers[2].weight.T, d=g["last_update"]["layers.2.weight"].T)

        ### set default lrs to try
        if lrs_to_try is None:
            lrs_to_try = []
            for t in (1/4, 1/2, 3/4, 1):
                for l in range(1, 8):
                    lrs_to_try.append(t * 10**(-l))

        ### save original/current loss
        init_loss = task["loss_fn"](y_hat=optee(task=task))

        ### find the largest decrease in loss to select lr
        best = {"loss_decrease": -np.inf, "lr": None}
        for lr in lrs_to_try:
            y_hat = optee.forward_w_params( # TODO: run all lrs at once
                params={
                    "layers.0.weight": optee.layers[0].weight - lr * g["last_update"]["layers.0.weight"],
                    "layers.0.bias": optee.layers[0].bias - lr * g["last_update"]["layers.0.bias"],
                    # "layers.2.weight": optee.layers[2].weight - lr * g["last_update"]["layers.2.weight"],
                },
                params_for=None,
                task=task,
            )
            loss = task["loss_fn"](y_hat=y_hat)

            if init_loss - loss > best["loss_decrease"]:
                best["loss_decrease"] = (init_loss - loss).item()
                best["lr"] = lr
        lr_out["layers.0.weight"] = lr_out["layers.0.bias"] = best["lr"]

    return lr_out


def parallel_n_step_lookahead_lr_search_hfunc_tanh_twolayer_optee(
    task,
    optee,
    g,
    n_steps=1,
    lrs_to_try=None,
):
    assert n_steps == 1, "More than one step lookahead not yet implemented."

    lr_out = dict()
    with torch.no_grad():
        ### set default lrs to try
        if lrs_to_try is None:
            lrs_to_try = []
            for t in (1/4, 1/2, 3/4, 1):
                for l in range(1, 8):
                    lrs_to_try.append(t * 10**(-l))

        ### save original/current loss
        init_loss = task["loss_fn"](y_hat=optee(task=task))

        ### find the largest decrease in loss to select lr
        best = {"loss_decrease": -np.inf, "lr": None}
        for lr in lrs_to_try:
            ### quadratic wrt to `layers.2.weight`
            first_layer_out = optee.forward_w_params(
                params={
                    "layers.0.weight": optee.layers[0].weight - lr * g["last_update"]["layers.0.weight"],
                    "layers.0.bias": optee.layers[0].bias - lr * g["last_update"]["layers.0.bias"],
                },
                params_for=None,
                stop_at_layer_idx=2, # stop before applying this weight matrix
                task=task,
            ) # (B, hidden_dim=N)
            A = first_layer_out.T @ first_layer_out # (N, N)
            b = -1 * first_layer_out.T @ task["y"] # (N)
            lr_layer_2 = get_optimal_lr(A=A, b=b, x=optee.layers[2].weight.T, d=g["last_update"]["layers.2.weight"].T)

            y_hat = optee.forward_w_params( # TODO: run all lrs at once
                params={
                    "layers.0.weight": optee.layers[0].weight - lr * g["last_update"]["layers.0.weight"],
                    "layers.0.bias": optee.layers[0].bias - lr * g["last_update"]["layers.0.bias"],
                    "layers.2.weight": optee.layers[2].weight - lr_layer_2 * g["last_update"]["layers.2.weight"],
                },
                params_for=None,
                task=task,
            )
            loss = task["loss_fn"](y_hat=y_hat)

            if init_loss - loss > best["loss_decrease"]:
                best["loss_decrease"] = (init_loss - loss).item()
                best["lr"] = lr
                best["lr_layer_2"] = lr_layer_2
        lr_out["layers.0.weight"] = lr_out["layers.0.bias"] = best["lr"]
        lr_out["layers.2.weight"] = best["lr_layer_2"]

    return lr_out


def per_param_n_step_lookahead_lr_search_hfunc_tanh_twolayer_optee(
    task,
    optee,
    g,
    n_steps=1,
    lrs_to_try=None,
):
    assert n_steps == 1, "More than one step lookahead not yet implemented."

    lr_out = dict()
    with torch.no_grad():
        ### set default lrs to try
        if lrs_to_try is None:
            lrs_to_try = []
            for t in (1/4, 1/2, 3/4, 1):
                for l in range(0, 8):
                    lrs_to_try.append(t * 10**(-l))

        ### save original/current loss
        init_loss = task["loss_fn"](y_hat=optee(task=task))

        ### find the largest decrease in loss to select per-param lr
        for n, p in optee.all_named_parameters():
            best = {"loss_decrease": -np.inf, "lr": None}
            for lr in lrs_to_try:
                y_hat = optee.forward_w_params( # TODO: run all lrs at once
                    params={
                        n: p - lr * g["last_update"][n],
                    },
                    params_for=None,
                    task=task,
                )
                loss = task["loss_fn"](y_hat=y_hat)

                if init_loss - loss > best["loss_decrease"]:
                    best["loss_decrease"] = (init_loss - loss).item()
                    best["lr"] = lr
            lr_out[n] = best["lr"]

    return lr_out
