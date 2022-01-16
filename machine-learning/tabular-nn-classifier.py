"""
# Tabular embedding

Tabular data includes both numeric and categorical variables and can include 
missing values. The embedding module encodes all inputs into quantile bins and 
encodes missing values as a special token that contiributes zero-weight. The 
tokenization process is performed on a per-column basis so that tokens are 
monotonically increasing and do not overlap across features (except for the 
missing value indicator token).

Binning is a common preprocessing step for tabular data and used in XGBoost, 
LightGBM, EBM, and other machine learning algorithms prior to fitting.

Training loss and validation AUC on the Adult dataset are:

```
Epoch: 0 | Train loss: 0.7412 | Val metric: -0.8773
Epoch: 1 | Train loss: 0.5579 | Val metric: -0.8841
Epoch: 2 | Train loss: 0.4866 | Val metric: -0.8874
Epoch: 3 | Train loss: 0.4502 | Val metric: -0.8893
Epoch: 4 | Train loss: 0.4283 | Val metric: -0.8905
Epoch: 5 | Train loss: 0.4136 | Val metric: -0.8913
Epoch: 6 | Train loss: 0.4038 | Val metric: -0.8919
Epoch: 7 | Train loss: 0.3956 | Val metric: -0.8924
Epoch: 8 | Train loss: 0.3895 | Val metric: -0.8927
Epoch: 9 | Train loss: 0.3846 | Val metric: -0.8930
```

Performance on the adult income dataset:

```
breakeven_thresh        -0.388
breakeven_precision      0.667
breakeven_recall         0.667
breakeven_specificity    0.893
auc                      0.896
spec90_thresh           -0.328
spec90_recall            0.648
spec90_precision         0.675
spec95_thresh            0.216
spec95_recall            0.488
spec95_precision         0.759
```
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from sklearn.metrics import roc_auc_score

import utils
import config
import datasets

importlib.reload(utils)
importlib.reload(config)
importlib.reload(datasets)


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim=1,
        num_classes=1,
        expected_value=0.0,
        padding_idx=0,
    ) -> None:
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.output = nn.Linear(embedding_dim, num_classes)
        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.constant_(self.output.bias, expected_value)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        emb = self.embedding(x)  # [B, H, 1]
        emb = emb.sum(dim=1)  # [B, 1]
        preds = self.output(emb)  # [B, 1]
        return preds.squeeze(-1)


@dataclass
class ClassifierOpts:
    num_embeddings: int
    embedding_dim: int
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


def train():
    data = datasets.load_adult()
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.20, random_state=config.SEED
    )

    quantile_transform = utils.QuantileBinningTransform(n_quantiles=11)
    quantile_transform.fit(X_train)
    X_train_binned = quantile_transform.transform(X_train)
    X_test_binned = quantile_transform.transform(X_test)

    opts = ClassifierOpts(
        num_embeddings=quantile_transform.vocab_size,
        embedding_dim=1,
        num_classes=1,
        expected_value=utils.get_baseline_prediction(y_train),
        lr=0.1,
        epochs=10,
    )

    # Model
    model = EmbeddingModel(
        num_embeddings=opts.num_embeddings,
        embedding_dim=opts.embedding_dim,
        num_classes=opts.num_classes,
        expected_value=opts.expected_value,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=opts.beta)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    losses = utils.AverageMeter()
    metrics = utils.AverageMeter()

    # data loaders
    train_loader = utils.create_generic_dataloader(
        (X_train_binned, y_train),
        batch_size=opts.batch_size,
        shuffle=True,
        device=opts.device,
        dtypes=["long", "float"],
    )
    test_loader = utils.create_generic_dataloader(
        (X_test_binned, y_test),
        batch_size=opts.batch_size,
        shuffle=False,
        device=opts.device,
        dtypes=["long", "float"],
    )

    for epoch in range(opts.epochs):
        run_epoch(model, opts, train_loader, optimizer, scheduler, losses, metrics, is_train=True)
        run_epoch(model, opts, test_loader, optimizer, scheduler, losses, metrics, is_train=False)
        print(f"Epoch: {epoch} | Train loss: {losses.avg:.4f} | Val metric: {metrics.avg: .4f}")

    y_preds = []
    for batch in test_loader:
        batch_x, _ = batch
        preds = model(batch_x)
        y_preds.append(preds)
    y_preds = torch.cat(y_preds).detach().cpu().numpy().ravel()

    from cdsutils.performance import performance
    perf = pd.Series(performance(y_test, y_preds))
    perf.round(3)
