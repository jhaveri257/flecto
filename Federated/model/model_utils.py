"""
Federated/model/model_utils.py
================================
Utility functions for checkpoint management and model-weight operations.

This module provides two complementary abstraction layers for working with
PyTorch model parameters:

**Flat-vector API** (``get_flat_params`` / ``set_flat_params``):
    Operates directly on the model's ``nn.Parameter`` tensors in-place.
    Used internally by the update selector and compressor during a training
    round, where the model is already loaded and we want zero-copy operations.

**State-dict API** (``state_dict_to_vector`` / ``vector_to_state_dict``):
    Converts between a ``state_dict`` (an ``OrderedDict`` of named tensors)
    and a single flat float32 numpy array.  The state-dict representation is
    the canonical serializable form used for checkpointing, communication
    payloads, and server-side reconstruction — it is completely decoupled from
    any live model instance.

Both APIs produce equivalent flat representations and are interchangeable;
the state-dict API is preferred for server-side operations where no model
instance exists.

Public API
----------
save_checkpoint(model, round_idx, filename, models_dir) -> str
load_checkpoint(model, path, device) -> int
clone_model(model) -> nn.Module
copy_weights(src, dst) -> None
get_flat_params(model) -> torch.Tensor
set_flat_params(model, flat_params) -> None
get_flat_param_count(model) -> int
state_dict_to_vector(state_dict) -> np.ndarray
vector_to_state_dict(vector, reference_state_dict) -> OrderedDict
"""

from __future__ import annotations

import os
import copy
from collections import OrderedDict
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

from Federated.config import MODELS_DIR


# ===========================================================================
# Checkpoint I/O
# ===========================================================================

def save_checkpoint(
    model: nn.Module,
    round_idx: int,
    filename: Optional[str] = None,
    models_dir: str = MODELS_DIR,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a model checkpoint to disk.

    The checkpoint file is a Python dict with at least two keys:
    ``"round"`` and ``"model_state_dict"``.  Additional metadata can be
    stored via ``extra_meta`` (e.g., accuracy, loss, round time).

    Parameters
    ----------
    model : nn.Module
        The PyTorch model whose ``state_dict`` will be saved.
    round_idx : int
        Current FL communication round number. Used to auto-generate the
        filename when ``filename`` is None.
    filename : str, optional
        Explicit output filename (e.g., ``"final_model.pt"``). When None,
        defaults to ``"global_model_round_{round_idx:04d}.pt"``.
    models_dir : str, optional
        Directory path where the checkpoint will be written.
        Created automatically if it does not exist.
    extra_meta : dict, optional
        Additional key-value pairs merged into the checkpoint dict.
        Useful for storing round metrics alongside the model weights.

    Returns
    -------
    str
        Absolute path to the written ``.pt`` file.

    Examples
    --------
    >>> path = save_checkpoint(model, round_idx=5)
    >>> path = save_checkpoint(model, round_idx=5,
    ...                        extra_meta={"accuracy": 0.93, "loss": 0.21})
    """
    os.makedirs(models_dir, exist_ok=True)

    if filename is None:
        filename = f"global_model_round_{round_idx:04d}.pt"

    path = os.path.join(models_dir, filename)

    payload: Dict[str, Any] = {
        "round": round_idx,
        "model_state_dict": model.state_dict(),
    }
    if extra_meta:
        payload.update(extra_meta)

    torch.save(payload, path)
    return path


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: Optional[torch.device] = None,
) -> int:
    """
    Load a checkpoint file and restore model weights in-place.

    Parameters
    ----------
    model : nn.Module
        Target model whose weights are overwritten with the checkpoint data.
        The model architecture must match the saved ``state_dict``.
    path : str
        Path to the ``.pt`` checkpoint file produced by ``save_checkpoint``.
    device : torch.device, optional
        Device to map tensors to when loading. Defaults to CPU.

    Returns
    -------
    int
        The ``"round"`` value stored in the checkpoint (0 if not present).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not point to an existing file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return int(checkpoint.get("round", 0))


# ===========================================================================
# Model State Cloning & Distribution
# ===========================================================================

def clone_model(model: nn.Module) -> nn.Module:
    """
    Return a fully independent deep copy of the model.

    Used by the server to distribute a snapshot of the global model to each
    simulated client at the start of every communication round, without
    sharing underlying parameter tensors.

    Parameters
    ----------
    model : nn.Module
        Source model to copy.

    Returns
    -------
    nn.Module
        A completely independent model instance with identical weights.
        Modifying the copy does not affect the original.
    """
    return copy.deepcopy(model)


def copy_weights(src: nn.Module, dst: nn.Module) -> None:
    """
    Copy all parameter values from ``src`` into ``dst`` in-place.

    Preferred over ``clone_model`` when ``dst`` already exists and we only
    want to reset its weights (avoids allocating a new model instance).

    Parameters
    ----------
    src : nn.Module
        Source model (read-only — not modified).
    dst : nn.Module
        Destination model (all weights overwritten in-place).
        Must have the same architecture as ``src``.
    """
    dst.load_state_dict(copy.deepcopy(src.state_dict()))


# ===========================================================================
# Flat Parameter Vector API
# ===========================================================================

def get_flat_params(model: nn.Module) -> torch.Tensor:
    """
    Concatenate all trainable parameter tensors into a single 1-D float32 vector.

    The concatenation order follows ``model.parameters()``, which is
    deterministic and consistent across calls for the same model architecture.
    This property is relied upon by ``set_flat_params`` for correct round-trips.

    Used by the update selector to rank **all** model parameters by absolute
    magnitude in a single global Top-K pass — avoiding the per-layer bias that
    would result from applying Top-K independently to each weight matrix.

    Parameters
    ----------
    model : nn.Module
        Source model.

    Returns
    -------
    torch.Tensor
        Flat float32 tensor of shape ``(total_trainable_params,)``.
    """
    return torch.cat(
        [p.data.view(-1) for p in model.parameters() if p.requires_grad]
    ).float()


def set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """
    Restore model parameters from a flat 1-D vector in-place.

    Reverses the concatenation performed by ``get_flat_params``.  The order
    of slices corresponds exactly to ``model.parameters()``.

    Parameters
    ----------
    model : nn.Module
        Target model (weights modified in-place).
    flat_params : torch.Tensor
        Flat float32 tensor of shape ``(total_trainable_params,)``.
        Must have the same length as ``get_flat_param_count(model)``.

    Raises
    ------
    RuntimeError
        If ``flat_params`` has a different number of elements than the model.
    """
    expected = get_flat_param_count(model)
    if flat_params.numel() != expected:
        raise RuntimeError(
            f"set_flat_params: size mismatch — "
            f"got {flat_params.numel()}, expected {expected}."
        )

    offset = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        numel = p.numel()
        p.data.copy_(flat_params[offset : offset + numel].view(p.shape))
        offset += numel


def get_flat_param_count(model: nn.Module) -> int:
    """
    Return the total number of trainable scalar parameters in a model.

    Parameters
    ----------
    model : nn.Module
        Source model.

    Returns
    -------
    int
        Sum of ``p.numel()`` over all parameters with ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# State-Dict Vector API  (Improvement 3)
# ===========================================================================

def state_dict_to_vector(state_dict: "OrderedDict[str, torch.Tensor]") -> np.ndarray:
    """
    Serialise a model ``state_dict`` into a flat float32 numpy array.

    This is the preferred representation for server-side operations (FedAvg
    aggregation, sparse reconstruction, error-feedback storage) because it
    does not require a live model instance.  The resulting vector can be
    stored in a numpy array, serialised to bytes, or operated on with
    standard numpy broadcasting.

    Only tensors that are floating-point (trainable weight/bias parameters)
    are included.  Non-floating-point buffers (e.g., ``num_batches_tracked``
    in BatchNorm) are skipped.

    The concatenation order is the insertion order of ``state_dict``, which
    matches ``model.state_dict()`` output order — deterministic for a given
    architecture.

    Parameters
    ----------
    state_dict : OrderedDict[str, torch.Tensor]
        A model's ``state_dict()`` as returned by ``model.state_dict()``.

    Returns
    -------
    np.ndarray
        Flat float32 array of shape ``(total_float_params,)``.

    See Also
    --------
    vector_to_state_dict : Inverse operation.
    get_flat_params : Equivalent operation operating on a live model.

    Examples
    --------
    >>> sd = model.state_dict()
    >>> vec = state_dict_to_vector(sd)
    >>> vec.shape
    (109386,)
    >>> vec.dtype
    dtype('float32')
    """
    parts = []
    for key, tensor in state_dict.items():
        if tensor.is_floating_point():
            parts.append(tensor.detach().cpu().view(-1).float().numpy())
    return np.concatenate(parts).astype(np.float32)


def vector_to_state_dict(
    vector: np.ndarray,
    reference_state_dict: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    """
    Reconstruct a model ``state_dict`` from a flat float32 numpy array.

    Reverses ``state_dict_to_vector``.  The ``reference_state_dict`` is used
    only to determine the name, shape, and dtype of each tensor; its values
    are overwritten.  Non-floating-point entries are carried through unchanged
    (copied from ``reference_state_dict``).

    Parameters
    ----------
    vector : np.ndarray
        Flat float32 array of shape ``(total_float_params,)`` as produced
        by ``state_dict_to_vector``.
    reference_state_dict : OrderedDict[str, torch.Tensor]
        A ``state_dict`` from any model with the same architecture.
        Provides the tensor shapes and insertion order needed to un-flatten
        ``vector`` back into the correct named tensors.

    Returns
    -------
    OrderedDict[str, torch.Tensor]
        A new ``state_dict`` with all floating-point tensors populated from
        ``vector`` and non-floating-point tensors copied from
        ``reference_state_dict``.  Can be loaded directly via
        ``model.load_state_dict(result)``.

    Raises
    ------
    ValueError
        If ``vector`` length does not match the total number of float
        parameters in ``reference_state_dict``.

    See Also
    --------
    state_dict_to_vector : Inverse operation.

    Examples
    --------
    >>> sd = model.state_dict()
    >>> vec = state_dict_to_vector(sd)
    >>> # Modify vec (e.g., apply aggregated update)
    >>> new_sd = vector_to_state_dict(vec, sd)
    >>> model.load_state_dict(new_sd)
    """
    # Validate total length
    expected_len = sum(
        t.numel()
        for t in reference_state_dict.values()
        if t.is_floating_point()
    )
    if vector.shape[0] != expected_len:
        raise ValueError(
            f"vector_to_state_dict: length mismatch — "
            f"got {vector.shape[0]}, expected {expected_len}."
        )

    new_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    offset = 0

    for key, ref_tensor in reference_state_dict.items():
        if ref_tensor.is_floating_point():
            numel = ref_tensor.numel()
            chunk = vector[offset : offset + numel]
            new_state_dict[key] = (
                torch.from_numpy(chunk.copy())
                .view(ref_tensor.shape)
                .to(dtype=ref_tensor.dtype)
            )
            offset += numel
        else:
            # Non-float buffers (e.g., running_mean in BN) — pass through
            new_state_dict[key] = ref_tensor.clone()

    return new_state_dict
