"""
Federated/model/fl_model.py
============================
Lightweight MLP for Federated Learning experiments on MNIST.

Architecture (Problem Statement 3, Requirement 2):

    Input  (784)
        |   Linear(784, 128) + ReLU + Dropout
       128
        |   Linear(128,  64) + ReLU + Dropout
        64
        |   Linear(64,   10)   <-- raw logits, NO activation
    Output  (10)

Design decisions:

    No CNN:
        The research objective is communication efficiency, not
        image classification accuracy. A flat MLP exposes all
        parameters symmetrically, making Top-K magnitude selection
        straightforward and unbiased across layers.

    Raw logit output (no LogSoftmax):
        ``nn.CrossEntropyLoss`` internally applies ``log_softmax``
        before the negative log-likelihood. Returning raw logits
        avoids the double application of softmax and is the
        standard PyTorch idiom recommended in the official docs.

    No BatchNorm:
        Batch statistics differ per client and are NOT aggregated
        by FedAvg. BatchNorm layers would introduce silent
        distribution mismatch after aggregation. LayerNorm or no
        normalization is preferred in FL (Li et al., 2021,
        "Federated Optimization in Heterogeneous Networks").

    Dropout:
        Rate is configurable via ``config.DROPOUT_RATE``.
        Helps regularize small, skewed client shards in Non-IID
        settings. Disabled automatically during evaluation via
        ``model.eval()``.

References:
    McMahan et al. (2017). "Communication-Efficient Learning of Deep
    Networks from Decentralized Data." AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

from __future__ import annotations

import torch
import torch.nn as nn

from Federated.config import (
    INPUT_DIM,
    NUM_CLASSES,
    HIDDEN_1,
    HIDDEN_2,
    DROPOUT_RATE,
)


class FLModel(nn.Module):
    """
    Lightweight fully-connected MLP for FL communication-efficiency research.

    Produces raw logits (unnormalised scores) for each of the ``num_classes``
    output classes.  These logits are passed directly to
    ``torch.nn.CrossEntropyLoss`` during training — no external softmax or
    log-softmax is needed.

    Architecture::

        Flatten -> Linear(input_dim, hidden_1) -> ReLU -> Dropout
                -> Linear(hidden_1, hidden_2)  -> ReLU -> Dropout
                -> Linear(hidden_2, num_classes)       [raw logits]

    Parameters
    ----------
    input_dim : int, optional
        Flattened input size. Default is ``INPUT_DIM`` (784 for MNIST 28x28).
    hidden_1 : int, optional
        Width of the first hidden layer. Default is ``HIDDEN_1`` (128).
    hidden_2 : int, optional
        Width of the second hidden layer. Default is ``HIDDEN_2`` (64).
    num_classes : int, optional
        Number of output logits. Default is ``NUM_CLASSES`` (10 for MNIST).
    dropout_rate : float, optional
        Dropout probability applied after each hidden layer activation.
        Default is ``DROPOUT_RATE`` (0.1).

    Attributes
    ----------
    input_dim : int
    hidden_1 : int
    hidden_2 : int
    num_classes : int
    net : nn.Sequential
        The full forward-pass computation graph.

    Examples
    --------
    >>> model = FLModel()
    >>> x = torch.randn(32, 1, 28, 28)   # MNIST batch
    >>> logits = model(x)                 # shape (32, 10)
    >>> loss = nn.CrossEntropyLoss()(logits, labels)
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_1: int = HIDDEN_1,
        hidden_2: int = HIDDEN_2,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Flatten(),                           # (B,1,28,28) -> (B,784)
            nn.Linear(input_dim, hidden_1),         # Hidden layer 1
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_1, hidden_2),          # Hidden layer 2
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_2, num_classes),       # Output: raw logits
            # NO LogSoftmax here — CrossEntropyLoss applies it internally.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass and return raw class logits.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 1, 28, 28)`` or ``(B, 784)``.

        Returns
        -------
        torch.Tensor
            Raw unnormalised logits of shape ``(B, num_classes)``.
            Pass directly to ``nn.CrossEntropyLoss`` — do NOT apply
            softmax before the loss function.
        """
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return predicted class indices (argmax of logits).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 1, 28, 28)`` or ``(B, 784)``.

        Returns
        -------
        torch.Tensor
            Integer class predictions of shape ``(B,)``.
        """
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)

    def count_parameters(self) -> int:
        """
        Count total trainable scalar parameters.

        Returns
        -------
        int
            Sum of ``p.numel()`` for all parameters where
            ``p.requires_grad is True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_size_bytes(self) -> int:
        """
        Return total parameter memory footprint in bytes.

        Assumes float32 precision (4 bytes per scalar).

        Returns
        -------
        int
            ``count_parameters() * 4``.
        """
        return self.count_parameters() * 4

    def __repr__(self) -> str:
        n_params = self.count_parameters()
        size_kb = self.parameter_size_bytes() / 1024
        return (
            f"FLModel(\n"
            f"  Input({self.input_dim}) -> Linear({self.input_dim},{self.hidden_1})"
            f" -> ReLU -> Dropout\n"
            f"  -> Linear({self.hidden_1},{self.hidden_2}) -> ReLU -> Dropout\n"
            f"  -> Linear({self.hidden_2},{self.num_classes}) [raw logits]\n"
            f"  Parameters: {n_params:,} ({size_kb:.1f} KB)\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    model = FLModel()
    print(model)

    # Forward pass smoke test
    x = torch.randn(16, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (16, 10), f"Unexpected shape: {logits.shape}"
    print(f"\n[PASS] Forward pass OK — logit shape: {logits.shape}")

    # CrossEntropyLoss compatibility check
    labels = torch.randint(0, 10, (16,))
    loss = nn.CrossEntropyLoss()(logits, labels)
    assert loss.item() > 0, "Loss must be positive"
    print(f"[PASS] CrossEntropyLoss OK — loss: {loss.item():.4f}")

    # predict() helper
    preds = model.predict(x)
    assert preds.shape == (16,), f"Unexpected predict shape: {preds.shape}"
    print(f"[PASS] predict() OK — shape: {preds.shape}")

    print(f"[PASS] Parameter count: {model.count_parameters():,}")
    print(f"[PASS] Model size: {model.parameter_size_bytes() / 1024:.1f} KB")
    sys.exit(0)
