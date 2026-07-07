"""
Federated/client/local_trainer.py
===================================
Client-side local training loop for Federated Learning.

Each FL client:

1. Receives the current global model from the server (distributed at round start).
2. Trains on its private local dataset for ``LOCAL_EPOCHS`` epochs using SGD.
3. Returns the trained model weights plus training diagnostics.

The server then computes the weight delta:

    delta_W = W_local - W_global

Design decisions
----------------
SGD with momentum:
    Standard FL baseline from McMahan et al. (2017).  Adam is intentionally
    avoided: its adaptive per-parameter learning rates interact poorly with
    FedAvg averaging and produce inconsistent convergence in Non-IID settings.

CrossEntropyLoss:
    Pairs with the raw-logit output of FLModel.  ``nn.CrossEntropyLoss``
    internally applies ``log_softmax`` then ``nll_loss``, which is numerically
    stable and the recommended PyTorch idiom.  The previous ``NLLLoss +
    LogSoftmax`` formulation was equivalent but non-standard.

Gradient clipping (max_norm=1.0):
    Prevents exploding gradients when client objectives diverge in Non-IID
    settings (client drift).  The clip norm is configurable via
    ``config.GRAD_CLIP_NORM``.

Stateless design:
    Each call to ``train()`` is fully self-contained.  The trainer holds
    no mutable round-level state between calls.

References
----------
McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks
from Decentralized Data." AISTATS 2017. https://arxiv.org/abs/1602.05629

Li et al. (2020). "Convergence of FedProx." ICLR 2020.
https://arxiv.org/abs/1812.06127
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Federated.config import (
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    MOMENTUM,
    WEIGHT_DECAY,
)


class LocalTrainer:
    """
    Stateless local SGD trainer for a single FL client.

    Each call to ``train()`` is independent: no internal state is mutated
    between rounds.  The trainer is safe to reuse across communication rounds
    and across multiple FL experiments in the same Python process.

    Parameters
    ----------
    client_id : int
        Integer identifier for this client.  Used in returned metrics for
        logging and debugging.
    device : torch.device
        Computation device (CPU or CUDA).
    local_epochs : int, optional
        Number of full passes over the local dataset per FL round.
        Default is ``LOCAL_EPOCHS`` from ``config.py``.
    lr : float, optional
        SGD learning rate. Default is ``LEARNING_RATE``.
    momentum : float, optional
        SGD momentum coefficient. Default is ``MOMENTUM``.
    weight_decay : float, optional
        L2 regularisation coefficient. Default is ``WEIGHT_DECAY``.
    grad_clip_norm : float, optional
        Maximum L2 norm for gradient clipping. Default is ``GRAD_CLIP_NORM``.

    Attributes
    ----------
    client_id : int
    device : torch.device
    local_epochs : int
    lr : float
    momentum : float
    weight_decay : float
    grad_clip_norm : float
    criterion : nn.CrossEntropyLoss
        The loss function used during training.
    """

    def __init__(
        self,
        client_id: int,
        device: torch.device,
        local_epochs: int = LOCAL_EPOCHS,
        lr: float = LEARNING_RATE,
        momentum: float = MOMENTUM,
        weight_decay: float = WEIGHT_DECAY,
        grad_clip_norm: float = GRAD_CLIP_NORM,
    ) -> None:
        self.client_id = client_id
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm

        # CrossEntropyLoss pairs with the raw-logit output of FLModel.
        # It internally applies log_softmax, so no activation is needed
        # on the model's output layer.
        self.criterion = nn.CrossEntropyLoss()

    def train(
        self,
        model: nn.Module,
        data_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Execute local SGD training and return weights plus diagnostics.

        The model is trained in-place.  A deep copy of the final
        ``state_dict`` is also returned so the caller can access the
        post-training weights without holding a reference to the model
        object.

        Parameters
        ----------
        model : nn.Module
            Local model copy (a clone of the global model for this round).
            Modified **in-place** during training.
        data_loader : DataLoader
            DataLoader wrapping this client's local data shard.

        Returns
        -------
        dict
            A dictionary containing:

            ``"state_dict"`` : OrderedDict
                Deep copy of ``model.state_dict()`` after training.
                Used by the server to compute the weight delta.

            ``"loss"`` : float
                Average cross-entropy loss over the **final** epoch.

            ``"avg_loss"`` : float
                Average cross-entropy loss averaged over **all** epochs.

            ``"epoch_losses"`` : list[float]
                Per-epoch average losses (length = ``local_epochs``).

            ``"accuracy"`` : float
                Classification accuracy on the local training data after
                the final epoch (fraction in [0, 1]).

            ``"num_samples"`` : int
                Total number of training samples processed across all epochs.

            ``"num_batches"`` : int
                Total number of mini-batches processed across all epochs.

            ``"epochs"`` : int
                Number of epochs completed (equals ``self.local_epochs``).

            ``"train_time_s"`` : float
                Wall-clock training time in seconds.

            ``"client_id"`` : int
                This client's identifier.

        Examples
        --------
        >>> trainer = LocalTrainer(client_id=0, device=torch.device("cpu"))
        >>> metrics = trainer.train(local_model, client_loader)
        >>> print(metrics["accuracy"], metrics["loss"])
        """
        model.to(self.device)
        model.train()

        optimizer = optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        epoch_losses: List[float] = []
        total_samples = 0
        total_batches = 0

        t_start = time.perf_counter()

        for _epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            for inputs, labels in data_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                logits = model(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()

                # Gradient clipping — prevents client drift in Non-IID settings
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip_norm)

                optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1
                total_samples += inputs.size(0)

            total_batches += epoch_batches
            epoch_losses.append(epoch_loss / max(1, epoch_batches))

        train_time_s = time.perf_counter() - t_start

        # --- Compute local accuracy on the training data (final model state) ---
        accuracy = self._compute_accuracy(model, data_loader)

        return {
            "state_dict":    copy.deepcopy(model.state_dict()),
            "loss":          epoch_losses[-1] if epoch_losses else 0.0,
            "avg_loss":      sum(epoch_losses) / max(1, len(epoch_losses)),
            "epoch_losses":  epoch_losses,
            "accuracy":      accuracy,
            "num_samples":   total_samples,
            "num_batches":   total_batches,
            "epochs":        self.local_epochs,
            "train_time_s":  train_time_s,
            "client_id":     self.client_id,
        }

    def _compute_accuracy(
        self,
        model: nn.Module,
        data_loader: DataLoader,
    ) -> float:
        """
        Compute classification accuracy on the local dataset.

        Called internally after training to report the final local accuracy.
        Switches the model to eval mode for inference and restores training
        mode before returning.

        Parameters
        ----------
        model : nn.Module
            The model to evaluate (not modified).
        data_loader : DataLoader
            DataLoader over this client's local data shard.

        Returns
        -------
        float
            Fraction of correctly classified samples in ``[0.0, 1.0]``.
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = model(inputs)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        model.train()
        return correct / max(1, total)
