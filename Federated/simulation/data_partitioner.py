"""
Federated/simulation/data_partitioner.py
==========================================
MNIST dataset loading and client data partitioning.

Supports two standard FL partitioning strategies, both of which are
referenced extensively in the FL literature:

IID (Independent and Identically Distributed)
    Each client receives an equal-sized random shard of the full training
    set.  Label classes are uniformly distributed across all clients.
    Provides the upper-bound performance baseline — unrealistic but useful
    for ablation studies.

Non-IID — Pathological (McMahan et al., 2017)
    Each client receives data from only ``N_CLASSES_PER_CLIENT`` distinct
    label classes.  The resulting label distribution is highly skewed and
    heterogeneous, which is more representative of real-world federated
    deployments (e.g., a mobile keyboard application where each user's
    typing style differs significantly).

Extended API
------------
Every partitioning function can optionally return a ``client_class_distribution``
matrix (shape: ``num_clients x num_classes``) that records how many samples
of each class each client owns.  This information is used for:

- Visualisation of Non-IID skew (heat-map or bar plots in the evaluator).
- Weighted aggregation experiments.
- Sanity-checking the Non-IID partitioning algorithm.

The distribution is computed by ``compute_class_distribution()`` and can
also be called independently on any list of index lists.

References
----------
McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks
from Decentralized Data." AISTATS 2017.
https://arxiv.org/abs/1602.05629
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from Federated.config import (
    BATCH_SIZE,
    DATASET_FRACTION,
    MNIST_DATA_DIR,
    N_CLASSES_PER_CLIENT,
    NUM_CLIENTS,
    PARTITION_MODE,
    PARTITION_SEED,
)


# ===========================================================================
# MNIST Loader
# ===========================================================================

def load_mnist(data_dir: str = MNIST_DATA_DIR) -> Tuple[Dataset, Dataset]:
    """
    Download (if necessary) and load the MNIST dataset.

    The training set is optionally subsampled by ``DATASET_FRACTION``
    to support rapid smoke-test runs.  The test set is always returned
    in full (10,000 samples).

    MNIST normalization:
    - Mean  = 0.1307
    - Std   = 0.3081

    These are the canonical per-pixel statistics computed over the full
    60,000-sample training set.

    Parameters
    ----------
    data_dir : str, optional
        Local filesystem path for the MNIST download cache.
        Default is ``MNIST_DATA_DIR`` from ``config.py``.

    Returns
    -------
    tuple[Dataset, Dataset]
        ``(train_dataset, test_dataset)`` — both are
        ``torchvision.datasets.MNIST`` instances with a ``ToTensor`` +
        ``Normalize`` transform applied.

    Raises
    ------
    ImportError
        If ``torchvision`` is not installed.

    Examples
    --------
    >>> train_ds, test_ds = load_mnist()
    >>> len(train_ds), len(test_ds)
    (60000, 10000)
    """
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for MNIST loading.\n"
            "Install with: pip install torchvision"
        ) from exc

    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    if DATASET_FRACTION < 1.0:
        rng = np.random.default_rng(PARTITION_SEED)
        n_keep = int(len(train_dataset) * DATASET_FRACTION)
        indices = rng.choice(len(train_dataset), size=n_keep, replace=False)
        train_dataset = Subset(train_dataset, indices.tolist())

    return train_dataset, test_dataset


# ===========================================================================
# IID Partitioning
# ===========================================================================

def partition_iid(
    dataset: Dataset,
    num_clients: int = NUM_CLIENTS,
    seed: int = PARTITION_SEED,
    return_distribution: bool = False,
) -> Tuple[List[List[int]], Optional[np.ndarray]]:
    """
    Partition dataset indices into equal-sized IID shards.

    Each client receives ``len(dataset) // num_clients`` samples drawn
    uniformly at random without replacement.  The last client absorbs any
    remainder samples.

    Parameters
    ----------
    dataset : Dataset
        Full training dataset (e.g., the MNIST training set).
    num_clients : int, optional
        Number of FL clients. Default is ``NUM_CLIENTS``.
    seed : int, optional
        RNG seed for reproducible partitioning. Default is ``PARTITION_SEED``.
    return_distribution : bool, optional
        When ``True``, also return the class distribution matrix.
        When ``False`` (default), the second return value is ``None``.

    Returns
    -------
    client_indices : list[list[int]]
        List of length ``num_clients``.  Each element is a list of integer
        indices into ``dataset`` assigned to that client.
    class_distribution : np.ndarray or None
        Array of shape ``(num_clients, num_classes)`` where entry
        ``[i, c]`` is the number of samples from class ``c`` owned by
        client ``i``.  ``None`` if ``return_distribution=False``.

    Examples
    --------
    >>> idx, dist = partition_iid(train_ds, num_clients=5,
    ...                           return_distribution=True)
    >>> dist.shape
    (5, 10)
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    indices = rng.permutation(n)

    shard_size = n // num_clients
    client_indices: List[List[int]] = []

    for i in range(num_clients):
        start = i * shard_size
        end = start + shard_size if i < num_clients - 1 else n
        client_indices.append(indices[start:end].tolist())

    distribution = None
    if return_distribution:
        distribution = compute_class_distribution(dataset, client_indices)

    return client_indices, distribution


# ===========================================================================
# Non-IID Partitioning (Pathological)
# ===========================================================================

def partition_non_iid(
    dataset: Dataset,
    num_clients: int = NUM_CLIENTS,
    n_classes_per_client: int = N_CLASSES_PER_CLIENT,
    seed: int = PARTITION_SEED,
    return_distribution: bool = False,
) -> Tuple[List[List[int]], Optional[np.ndarray]]:
    """
    Partition dataset into pathologically Non-IID shards.

    Strategy (McMahan et al., 2017):

    1. Sort all samples by class label.
    2. Divide into ``num_clients * n_classes_per_client`` equal-size shards.
    3. Shuffle the shard list.
    4. Assign ``n_classes_per_client`` shards to each client.

    The result: each client owns data from at most ``n_classes_per_client``
    distinct classes, producing severe label heterogeneity — the canonical
    stress-test for FL algorithms.

    Parameters
    ----------
    dataset : Dataset
        Full training dataset.
    num_clients : int, optional
        Number of FL clients. Default is ``NUM_CLIENTS``.
    n_classes_per_client : int, optional
        Number of distinct label classes per client.
        McMahan et al. (2017) use 2.  Higher values approach IID.
        Default is ``N_CLASSES_PER_CLIENT``.
    seed : int, optional
        RNG seed. Default is ``PARTITION_SEED``.
    return_distribution : bool, optional
        When ``True``, also return the class distribution matrix.

    Returns
    -------
    client_indices : list[list[int]]
        Partition as described above.
    class_distribution : np.ndarray or None
        Shape ``(num_clients, num_classes)`` when requested; else ``None``.

    Examples
    --------
    >>> idx, dist = partition_non_iid(train_ds, num_clients=10,
    ...                                n_classes_per_client=2,
    ...                                return_distribution=True)
    >>> # Most entries in dist should be 0 for Non-IID
    >>> (dist == 0).mean()  # high fraction of zeros expected
    """
    rng = np.random.default_rng(seed)

    labels = _extract_labels(dataset)
    n = len(labels)

    sorted_indices = np.argsort(labels, kind="stable")

    num_shards = num_clients * n_classes_per_client
    shard_size = max(1, n // num_shards)
    actual_num_shards = min(num_shards, n // shard_size)

    shards: List[np.ndarray] = []
    for i in range(actual_num_shards):
        start = i * shard_size
        end = start + shard_size if i < actual_num_shards - 1 else n
        shards.append(sorted_indices[start:end])

    shard_ids = rng.permutation(len(shards))
    shards_per_client = len(shards) // num_clients

    client_indices: List[List[int]] = []
    for i in range(num_clients):
        assigned: List[int] = []
        for j in range(shards_per_client):
            shard_id = shard_ids[i * shards_per_client + j]
            assigned.extend(shards[shard_id].tolist())
        client_indices.append(assigned)

    distribution = None
    if return_distribution:
        distribution = compute_class_distribution(dataset, client_indices)

    return client_indices, distribution


# ===========================================================================
# Class Distribution Helper  (Improvement 4)
# ===========================================================================

def compute_class_distribution(
    dataset: Dataset,
    client_indices: List[List[int]],
) -> np.ndarray:
    """
    Compute the per-client, per-class sample count matrix.

    Iterates over each client's index list and counts how many samples
    of each class label that client owns.

    Parameters
    ----------
    dataset : Dataset
        The full training dataset from which ``client_indices`` were drawn.
    client_indices : list[list[int]]
        Partition produced by ``partition_iid`` or ``partition_non_iid``.
        Each inner list contains the integer indices of one client's shard.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(num_clients, num_classes)`` where
        ``result[i, c]`` is the number of samples from class ``c``
        owned by client ``i``.

    Notes
    -----
    The number of classes is inferred automatically from the global label
    set.  For MNIST this is always 10.

    Examples
    --------
    >>> dist = compute_class_distribution(train_ds, client_indices)
    >>> dist.shape
    (10, 10)
    >>> dist.sum(axis=1)          # total samples per client
    array([6000, 6000, ...])
    >>> dist.sum(axis=0)          # total samples per class
    array([5923, 6742, ...])
    """
    all_labels = _extract_labels(dataset)
    num_classes = int(all_labels.max()) + 1
    num_clients = len(client_indices)

    distribution = np.zeros((num_clients, num_classes), dtype=np.int64)
    for client_id, indices in enumerate(client_indices):
        if len(indices) == 0:
            continue
        client_labels = all_labels[np.array(indices, dtype=np.int64)]
        for label in client_labels:
            distribution[client_id, int(label)] += 1

    return distribution


# ===========================================================================
# DataLoader Factory
# ===========================================================================

def make_client_loaders(
    dataset: Dataset,
    client_indices: List[List[int]],
    batch_size: int = BATCH_SIZE,
) -> List[DataLoader]:
    """
    Build a per-client DataLoader from a list of index partitions.

    Parameters
    ----------
    dataset : Dataset
        The full training dataset.
    client_indices : list[list[int]]
        Partition produced by ``partition_iid`` or ``partition_non_iid``.
    batch_size : int, optional
        Mini-batch size for local training. Default is ``BATCH_SIZE``.

    Returns
    -------
    list[DataLoader]
        One DataLoader per client.  Each DataLoader wraps only the indices
        assigned to that client.  Shuffling is enabled so that each epoch
        presents data in a different order.
    """
    loaders = []
    for indices in client_indices:
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,      # 0 for Windows compatibility (no fork)
            pin_memory=False,
        )
        loaders.append(loader)
    return loaders


def make_test_loader(
    test_dataset: Dataset,
    batch_size: int = 256,
) -> DataLoader:
    """
    Build a DataLoader for the shared global test set.

    Parameters
    ----------
    test_dataset : Dataset
        MNIST test split (10,000 samples).
    batch_size : int, optional
        Evaluation batch size. Default is 256.

    Returns
    -------
    DataLoader
        Non-shuffled DataLoader over the full test set.
    """
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


# ===========================================================================
# High-Level Partition Builder
# ===========================================================================

def build_partition(
    train_dataset: Dataset,
    mode: str = PARTITION_MODE,
    num_clients: int = NUM_CLIENTS,
    batch_size: int = BATCH_SIZE,
    return_distribution: bool = False,
) -> Tuple[List[DataLoader], List[int], Optional[np.ndarray]]:
    """
    Convenience function that partitions, builds DataLoaders, and optionally
    computes the class distribution in one call.

    Parameters
    ----------
    train_dataset : Dataset
        Full MNIST training dataset.
    mode : str, optional
        Partitioning mode.  ``"iid"`` or ``"non_iid"``.
        Default is ``PARTITION_MODE`` from ``config.py``.
    num_clients : int, optional
        Number of FL clients. Default is ``NUM_CLIENTS``.
    batch_size : int, optional
        Local mini-batch size for DataLoaders. Default is ``BATCH_SIZE``.
    return_distribution : bool, optional
        Whether to compute and return the class distribution matrix.
        Default is ``False``.

    Returns
    -------
    client_loaders : list[DataLoader]
        One DataLoader per client.
    client_sizes : list[int]
        Number of samples in each client's shard.
        Used by FedAvg for weighted aggregation.
    class_distribution : np.ndarray or None
        Shape ``(num_clients, num_classes)`` if ``return_distribution=True``,
        else ``None``.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"iid"`` or ``"non_iid"``.

    Examples
    --------
    >>> loaders, sizes, dist = build_partition(
    ...     train_ds, mode="non_iid", return_distribution=True
    ... )
    >>> len(loaders)
    10
    >>> dist.shape
    (10, 10)
    """
    if mode == "iid":
        client_indices, distribution = partition_iid(
            train_dataset, num_clients, return_distribution=return_distribution
        )
    elif mode == "non_iid":
        client_indices, distribution = partition_non_iid(
            train_dataset, num_clients, return_distribution=return_distribution
        )
    else:
        raise ValueError(
            f"Unknown partition mode: '{mode}'. Choose 'iid' or 'non_iid'."
        )

    client_loaders = make_client_loaders(train_dataset, client_indices, batch_size)
    client_sizes = [len(idx) for idx in client_indices]

    return client_loaders, client_sizes, distribution


# ===========================================================================
# Internal Utilities
# ===========================================================================

def _extract_labels(dataset: Dataset) -> np.ndarray:
    """
    Extract integer class labels from any Dataset or Subset instance.

    Handles the following cases:

    - ``torchvision.datasets.MNIST`` — exposes ``.targets`` as a
      ``torch.Tensor``.
    - ``torch.utils.data.Subset`` — recursively extracts from the
      underlying dataset and applies the index mask.
    - Any other dataset — falls back to iterating the dataset
      (slow but universally correct).

    Parameters
    ----------
    dataset : Dataset
        Any PyTorch dataset instance.

    Returns
    -------
    np.ndarray
        1-D integer array of class labels, length ``len(dataset)``.
    """
    from torch.utils.data import Subset as TorchSubset

    if isinstance(dataset, TorchSubset):
        all_labels = _extract_labels(dataset.dataset)
        return all_labels[np.array(dataset.indices, dtype=np.int64)]

    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.numpy().astype(np.int64)
        return np.array(targets, dtype=np.int64)

    # Slow universal fallback
    labels = []
    for _, y in dataset:
        labels.append(int(y.item()) if isinstance(y, torch.Tensor) else int(y))
    return np.array(labels, dtype=np.int64)


# ===========================================================================
# Self-Test
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Data Partitioner Self-Test")
    print("=" * 60)

    print("\nLoading MNIST...")
    train_ds, test_ds = load_mnist()
    print(f"  Train size: {len(train_ds):,}")
    print(f"  Test  size: {len(test_ds):,}")

    print(f"\nIID Partition ({NUM_CLIENTS} clients):")
    iid_idx, iid_dist = partition_iid(train_ds, NUM_CLIENTS,
                                       return_distribution=True)
    for i, idx in enumerate(iid_idx):
        print(f"  Client {i:2d}: {len(idx):5d} samples")
    print(f"  Class distribution shape: {iid_dist.shape}")

    print(f"\nNon-IID Partition ({NUM_CLIENTS} clients, "
          f"{N_CLASSES_PER_CLIENT} classes/client):")
    noniid_idx, noniid_dist = partition_non_iid(
        train_ds, NUM_CLIENTS, N_CLASSES_PER_CLIENT, return_distribution=True
    )
    labels_all = _extract_labels(train_ds)
    for i, idx in enumerate(noniid_idx):
        unique_cls = np.unique(labels_all[idx])
        print(f"  Client {i:2d}: {len(idx):5d} samples | "
              f"classes: {unique_cls}")

    print("\n[PASS] Data partitioner self-test complete.")
