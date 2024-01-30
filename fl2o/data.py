import os
import random

import numpy as np
import torch
from torchvision import datasets, transforms

DATA_PATH = os.path.join(os.getenv("DATA_PATH"))
DEFAULT_SEED = 0


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MNIST:
    def __init__(
        self,
        training=True,
        batch_size=128,
        only_classes=None,
        preload=False,
        standardize=True,
        seed=DEFAULT_SEED,
        device="cpu",
    ):
        ### set seed
        g = torch.Generator()
        g.manual_seed(seed)

        ### init dataset and loader
        trans = [transforms.ToTensor()]
        if standardize:
            trans.append(transforms.Normalize((0.1307,), (0.3081,)))
        dataset = datasets.MNIST(
            DATA_PATH,
            train=training,
            download=False,
            transform=transforms.Compose(trans),
        )

        if only_classes is not None:
            idx = torch.isin(
                dataset.targets,
                only_classes
                if type(only_classes) == torch.Tensor
                else torch.tensor(only_classes),
            )
            dataset.targets = dataset.targets[idx]
            dataset.data = dataset.data[idx]

        self.training = training
        self.only_classes = only_classes
        self.batch_size = batch_size
        self.seed = seed
        self.device = device

        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        self.batches = []
        self.curr_batch = 0

        if preload:
            self.preload_batches()

    def count_class_samples(self):
        """Counts the number of samples for each class in the dataset, returns a dictionary."""
        counts = {i: 0 for i in range(10)}
        for batch in self.loader:
            for label in batch[1]:
                counts[label.item()] += 1
        return counts

    def preload_batches(self, clear_batches=True):
        if clear_batches:
            self.batches = []
            self.curr_batch = 0
        for b in self.loader:
            self.batches.append(b)

    def sample(self):
        if self.curr_batch >= len(self.batches):
            self.preload_batches(clear_batches=True)
        batch = [self.batches[self.curr_batch][0].to(self.device), self.batches[self.curr_batch][1].to(self.device)]
        self.curr_batch += 1
        return {
            "x": batch[0],
            "y": batch[1],
            "loss_fn": lambda y_hat: torch.nn.functional.cross_entropy(
                y_hat, batch[1]
            ),
            "loss_fn_ext": torch.nn.CrossEntropyLoss(),
            "loss_fn_cls": torch.nn.CrossEntropyLoss,
        }


class CustomTask:
    def __init__(self, task, task_config):
        self.task = task
        self.task_config = task_config

        if hasattr(task, "__call__"):
            self.task = task(**task_config)

    def sample(self):
        return self.task


def generate_least_squares_task(d, m, verbose=True, device="cpu"):
    ### Least squares problem
    W = torch.randn(d, m, device=device) / m
    y = torch.randn(1, m, device=device)
    loss_fn = lambda y_hat: (y_hat @ W - y).pow(2).sum().mul(0.5)

    A = W @ W.T
    b = -W @ y.T

    ### Tikhonov regularization solution
    R = torch.eye(d, device=device)
    for i in range(d):
        R[i, i] = W[i, :].norm(p=2)
    # R = W.norm(p=2, dim=-1).diag()

    def x_tik_solution(gamma, c):
        x_bar = c * torch.ones(1, d, device=device)
        return (
            x_bar.T
            + (W @ W.T + gamma * R @ R.T).inverse()
            @ W @ (y.T - W.T @ x_bar.T)
        ).T

    def tik_problem_loss_fn(y_hat, gamma, c):
        x_bar = c * torch.ones(d, 1, device=device)
        return (
            torch.norm(
                W.T @ y_hat.T - y.T,
                p=2,
            )**2
            + gamma * torch.norm(
                R.T @ (y_hat.T - x_bar),
                p=2,
            )**2
        )

    ### Solution w/out regularization
    x_solution = ((W @ W.T).inverse() @ W @ y.T).T  # torch.linalg.solve(W.T, y.T).T

    if verbose:
        print(
            f"\nLeast squares problem generated:"
            + f"\n  | {d=}, {m=}"
            + f"\n  | Condition number of W @ W.T: {torch.linalg.cond(A).item():.4f}"
            + f"\n  -----"
            + f"\n  | x_tik* error (to Tik. regularized problem):"
            + f"\n      gamma=1.:  " + str(tik_problem_loss_fn(y_hat=x_tik_solution(gamma=1, c=1), gamma=1, c=1).item())
            + f"\n      gamma=10.:  " + str(tik_problem_loss_fn(y_hat=x_tik_solution(gamma=10, c=1), gamma=10, c=1).item())
            + f"\n  | x_tik* error (to the original problem):"
            + f"\n      gamma=1.:  " + str(loss_fn(y_hat=x_tik_solution(gamma=1., c=1.)).item())
            + f"\n      gamma=10.:  " + str(loss_fn(y_hat=x_tik_solution(gamma=10., c=1.)).item())
            + f"\n  | x* error:\n      {loss_fn(y_hat=x_solution).item():.4f}"
            + f"\n"
        )

    return {
        "W": W,
        "y": y,
        "loss_fn": loss_fn,
        "A": A,
        "b": b,
        "x_tik_solution": x_tik_solution,
        "tik_problem_loss_fn": tik_problem_loss_fn,
        "x_solution": x_solution,
    }