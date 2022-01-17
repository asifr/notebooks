"""
# Neural adaptive regression splines (NARS)

Performance on the adult income dataset:

```
breakeven_thresh        -1.441
breakeven_precision      0.702
breakeven_recall         0.702
breakeven_specificity    0.904
auc                      0.915
spec90_thresh           -1.486
spec90_recall            0.711
spec90_precision         0.696
spec95_thresh           -0.863
spec95_recall            0.566
spec95_precision         0.784
```
"""

from typing import Tuple, List
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import math
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import config
import datasets

importlib.reload(utils)
importlib.reload(config)
importlib.reload(datasets)


class RegressionSpline(nn.Module):
    """
    Linear basis splines with uniformly placed knots. Expects data in
    range [0,1]. Uses ReLU to fit local functions on parts of the data.

    Parameters:
    -----------
    n_features : int
        Number of features
    n_knots : int
        Number of knots, default 25
    kmin : int
        Minimum value of x, default 0
    kmax : int
        Maximum value of x, default 1
    device : str
        Device to use, default "cpu"
    """

    def __init__(
        self,
        n_features: int,
        n_knots: int = 25,
        kmin: int = 0,
        kmax: int = 1,
        device: str = "cpu",
    ):
        super(RegressionSpline, self).__init__()
        self.device = device
        self.n_features = n_features
        self.n_knots = n_knots
        self.kmin = kmin
        self.kmax = kmax
        self.knots = (
            torch.linspace(kmin, kmax, steps=self.n_knots)
            .unsqueeze(0)
            .repeat(n_features, 1)
            .to(device)
        )
        self.register_parameter(
            "left_weight",
            nn.Parameter(torch.Tensor(1, n_features, self.n_knots), requires_grad=True),
        )
        self.register_parameter(
            "right_weight",
            nn.Parameter(torch.Tensor(1, n_features, self.n_knots), requires_grad=True),
        )
        nn.init.xavier_normal_(self.left_weight)
        nn.init.xavier_normal_(self.right_weight)
        self.scale = torch.FloatTensor([math.sqrt(2*n_knots)])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        ReLU functions are used to construct piecewise linear segments between
        knot points. Weights scale the linear segments.

        Parameters:
        -----------
        x : torch.FloatTensor
            design matrix, in range [0,1], shape [batch_size, n_features]

        Returns:
        --------
        fx : torch.FloatTensor
            output, shape [batch_size, n_features]
        """
        xc = x.unsqueeze(2)
        xr = self.right_weight * torch.relu(xc - self.knots)
        xl = self.left_weight * torch.relu(self.knots - xc)
        xh = torch.sum(xl + xr, -1) / self.scale  # [batch_size, n_features]
        return xh

    def basis(self, steps: int = 100, device: str = None):
        """
        Risk curves (x, f(x)) for each feature

        Returns
        -------
        x, fx: torch.FloatTensor, torch.FloatTensor
            Return the risk `f(x)` for `x` uniformly spaced feature values
            in the range [kmin, kmax]
        """
        device = self.device if device is None else device
        x = (
            torch.linspace(self.kmin, self.kmax, steps)
            .unsqueeze(1)
            .repeat(1, self.n_features)
            .to(device)
        )
        fx = self.forward(x).to(device)
        return x, fx


class NARS(nn.Module):
    def __init__(
        self,
        input_size,
        expected_value=0.0,
        device="cpu",
    ) -> None:
        super(NARS, self).__init__()
        self.reg = RegressionSpline(
            input_size,
        )
        self.expected_value = torch.FloatTensor([expected_value]).to(device)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        emb = self.reg(x)  # [B, D]
        preds = emb.sum(dim=-1) + self.expected_value  # [B, ]
        return preds


@dataclass
class ClassifierOpts:
    input_size: int
    num_classes: int = 1
    expected_value: float = 0
    device: str = "cpu"
    batch_size: int = 1024
    lr: float = 0.001
    epochs: int = 10
    dropout: float = 0.0
    beta: Tuple[float] = (0.9, 0.99)
    grad_norm_clip: float = 6.0
    lambda_recon: float = 1e-4


def run_epoch(
    model, opts, loader, optimizer, scheduler, losses, metrics, is_train=True
):
    if is_train:
        model.train()
    else:
        model.eval()
    for batch in loader:
        batch_x, batch_y = batch
        preds = model(batch_x)

        n = preds.size(0)
        if is_train:
            loss = F.binary_cross_entropy_with_logits(preds, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.grad_norm_clip)
            optimizer.step()
            losses.update(loss.cpu().item(), n)
        else:
            y_score = preds.detach().cpu().numpy().ravel()
            y_true = batch_y.detach().cpu().numpy().ravel()
            auc = torch.FloatTensor([-roc_auc_score(y_true, y_score)])
            metrics.update(auc.cpu().item(), n)

    if not is_train:
        scheduler.step(metrics.avg)


@dataclass
class SaveOutput:
    opts: ClassifierOpts
    model: nn.Module
    optimizer: torch.optim.Optimizer
    quantiles: List[np.ndarray]


def train():
    # %run machine-learning/nars.py
    data = datasets.load_adult()
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.20, random_state=config.SEED
    )

    numeric_inds = [i for i, f in enumerate(data.variables) if f in data.numeric_vars]
    amin, amax = np.percentile(X_train[:, numeric_inds], [1,99], axis=0)
    X_train[:, numeric_inds] = (X_train[:, numeric_inds] - amin) / (amax - amin)
    X_test[:, numeric_inds] = (X_test[:, numeric_inds] - amin) / (amax - amin)

    opts = ClassifierOpts(
        input_size=X_train.shape[1],
        expected_value=utils.get_baseline_prediction(y_train),
        lr=0.1,
        epochs=10,
    )

    # Model
    model = NARS(
        input_size=opts.input_size,
        expected_value=opts.expected_value,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=opts.beta)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    losses = utils.AverageMeter()
    metrics = utils.AverageMeter()

    # data loaders
    train_loader = utils.create_generic_dataloader(
        (X_train, y_train),
        batch_size=opts.batch_size,
        shuffle=True,
        device=opts.device,
        dtypes=["float", "float"],
    )
    test_loader = utils.create_generic_dataloader(
        (X_test, y_test),
        batch_size=opts.batch_size,
        shuffle=False,
        device=opts.device,
        dtypes=["float", "float"],
    )

    for epoch in range(opts.epochs):
        run_epoch(
            model,
            opts,
            train_loader,
            optimizer,
            scheduler,
            losses,
            metrics,
            is_train=True,
        )
        run_epoch(
            model,
            opts,
            test_loader,
            optimizer,
            scheduler,
            losses,
            metrics,
            is_train=False,
        )
        print(
            f"Epoch: {epoch} | Train loss: {losses.avg:.4f} | Val metric: {metrics.avg: .4f}"
        )

    # predict
    y_preds = []
    for batch in test_loader:
        batch_x, _ = batch
        preds = model(batch_x)
        y_preds.append(preds)
    y_preds = torch.cat(y_preds).detach().cpu().numpy().ravel()

    from cdsutils.performance import performance
    import pandas as pd

    perf = pd.Series(performance(y_test, y_preds))
    perf.round(3)

    x, y = utils.torch_to_numpy(*model.reg.basis())

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.reset_defaults()

    # plot risk distribution
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plt.hist([
        y_preds[y_test == 0],
        y_preds[y_test == 1]
    ], density=True, label=["<=50", ">50K"], bins=40)
    utils.labels(ax, xlabel='Risk', ylabel="PDF")
    plt.tight_layout()
    plt.show(block=False)

    # plot risk curves
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    i = 3
    if i in numeric_inds:
        xi = x[:,i] * (amax[i] - amin[i]) - amin[i]
        yi = y[:,i]
    else:
        xi = x[[0,-1],i]
        yi = y[[0,-1],i]
    ax.plot(xi, yi, "-", lw=2)
    ax.axhline(0, color="k", lw=1)
    plt.title(data.variables[i])
    utils.labels(ax, xlabel=data.variables[i], ylabel="Risk")
    plt.tight_layout()
    plt.show(block=False)