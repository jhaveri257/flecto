"""
FC/fountain_code.py
====================
Random Linear Fountain Code over GF(2) --- Encoder & Decoder.

Design rationale:
  LT (Luby Transform) codes require k >> 1000 for the Robust Soliton
  degree distribution to produce a reliably peelable bipartite graph.
  For our use case (FL model update chunks, k = 64--256), LT codes
  require 5--7x redundancy, which defeats the research objective.

  Random Linear Fountain Codes over GF(2) achieve near-optimal overhead
  for ANY k.  Each encoded packet is a random XOR-combination of source
  symbols.  Decoding uses Gaussian elimination over GF(2).

  For n = k + O(log k) received packets, decode probability > 99.9%.
  For k=128, ratio ~1.15 already gives near-perfect decode.

Theory:
  Given n random binary vectors in GF(2)^k, the probability of full
  rank is:  P = prod_{i=0}^{k-1} (1 - 2^{-(n-i)})
  For n = k + 20 (any k): P > 0.999999.

References:
  MacKay, D.J.C. (2005). "Fountain codes." IEE Proc. Comm.
  Shokrollahi, A. (2006). "Raptor Codes." IEEE Trans. Inf. Theory.

Usage:
    from FC.fountain_code import FountainCode

    fc = FountainCode(k=128, seed=42)
    source   = np.random.randint(0, 256, size=128, dtype=np.uint8)
    encoded  = fc.encode(source, redundancy_ratio=1.3)
    received = simulate_erasure_channel(encoded, loss_rate=0.10)
    decoded  = fc.decode(received, k=128)    # np.ndarray or None
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Encoded Packet
# ---------------------------------------------------------------------------

class EncodedPacket:
    """
    A single fountain-encoded packet.

    Attributes:
        index  : Sequential packet index (reproduces coefficients from seed).
        coeffs : Binary coefficient vector, shape (k,), dtype=np.uint8.
                 coeffs[i] == 1 means source symbol i is included in the XOR.
        data   : XOR of the selected source symbols (a single uint8 value
                 for scalar symbols, or a 1-D array for block symbols).
    """
    __slots__ = ("index", "coeffs", "data")

    def __init__(self, index: int, coeffs: np.ndarray, data: np.ndarray):
        self.index = index
        self.coeffs = coeffs       # shape (k,), binary
        self.data = data            # shape () or (block_size,)

    @property
    def degree(self) -> int:
        """Number of source symbols XOR-combined in this packet."""
        return int(self.coeffs.sum())

    def __repr__(self) -> str:
        return f"EncodedPacket(idx={self.index}, deg={self.degree})"


# ---------------------------------------------------------------------------
# Fountain Code --- Encoder
# ---------------------------------------------------------------------------

class FountainCode:
    """
    Random Linear Fountain Code over GF(2).

    The RL agent controls two parameters per transmission round:
      - k               : number of source symbols (FL model chunk count)
      - redundancy_ratio: n = ceil(k * redundancy_ratio) encoded packets

    Args:
        k    : Source block size.
        seed : Master RNG seed for reproducible encoding.
    """

    def __init__(self, k: int, seed: int = 42):
        self.k = k
        self.seed = seed

    def _make_coefficients(self, packet_index: int) -> np.ndarray:
        """
        Generate binary coefficient vector for packet `packet_index`.

        Uses a per-packet RNG seeded deterministically so the receiver
        can reproduce the coefficient vector without side-channel info.

        The coefficient density is approximately 50% (each bit is i.i.d.
        Bernoulli(0.5)). This gives near-optimal rank probability for GF(2).
        """
        rng = np.random.default_rng(
            (self.seed * (2**20) + packet_index) & 0xFFFF_FFFF_FFFF_FFFF
        )
        coeffs = rng.integers(0, 2, size=self.k, dtype=np.uint8)
        # Ensure at least one coefficient is 1 (avoid all-zero rows)
        if coeffs.sum() == 0:
            coeffs[rng.integers(0, self.k)] = 1
        return coeffs

    def encode(
        self,
        source: np.ndarray,
        redundancy_ratio: float = 1.3,
    ) -> List[EncodedPacket]:
        """
        Encode k source symbols into n = ceil(k * redundancy_ratio) packets.

        Args:
            source           : 1-D numpy array of k source symbols (uint8).
            redundancy_ratio : Overhead factor >= 1.0.

        Returns:
            List of EncodedPacket objects.
        """
        if len(source) != self.k:
            raise ValueError(f"Source length {len(source)} != k={self.k}")

        n = math.ceil(self.k * redundancy_ratio)
        packets: List[EncodedPacket] = []

        for i in range(n):
            coeffs = self._make_coefficients(i)
            # XOR all source symbols where coeffs[j] == 1
            selected = source[coeffs == 1]
            if len(selected) == 0:
                data = np.uint8(0)
            else:
                data = selected[0]
                for s in selected[1:]:
                    data = np.uint8(int(data) ^ int(s))
            packets.append(EncodedPacket(index=i, coeffs=coeffs, data=np.atleast_1d(data)))

        return packets

    def decode(
        self,
        received: List[EncodedPacket],
        k: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Gaussian elimination decoder over GF(2).

        Builds the n x k binary coefficient matrix and the n-length
        data vector, then performs row reduction to solve for the k
        source symbols.

        Args:
            received : Subset of encoded packets (after erasure channel).
            k        : Number of source symbols. Defaults to self.k.

        Returns:
            np.ndarray of shape (k,) if decode succeeds, else None.
        """
        k = k if k is not None else self.k
        n = len(received)

        if n < k:
            return None   # Underdetermined system --- cannot decode

        # Build augmented matrix [A | b] over GF(2)
        # A is n x k (binary), b is n x 1 (uint8 data values)
        A = np.zeros((n, k), dtype=np.uint8)
        b = np.zeros(n, dtype=np.uint8)

        for row, pkt in enumerate(received):
            A[row, :] = pkt.coeffs
            b[row] = int(pkt.data.flat[0])

        # --- Gaussian elimination over GF(2) with partial pivoting ---
        pivot_row = 0
        pivot_cols = []   # column order of pivots

        for col in range(k):
            # Find a row with a 1 in this column at or below pivot_row
            found = -1
            for r in range(pivot_row, n):
                if A[r, col] == 1:
                    found = r
                    break

            if found == -1:
                continue   # No pivot in this column --- skip (rank deficient)

            # Swap rows
            if found != pivot_row:
                A[[pivot_row, found]] = A[[found, pivot_row]]
                b[pivot_row], b[found] = b[found], b[pivot_row]

            # Eliminate all other 1s in this column
            for r in range(n):
                if r != pivot_row and A[r, col] == 1:
                    A[r] ^= A[pivot_row]          # GF(2) row addition
                    b[r] = np.uint8(int(b[r]) ^ int(b[pivot_row]))

            pivot_cols.append(col)
            pivot_row += 1

            if pivot_row >= n:
                break

        if len(pivot_cols) < k:
            return None   # Rank < k --- cannot decode

        # Read off solution: after elimination, A is identity on pivot columns
        result = np.zeros(k, dtype=np.uint8)
        for i, col in enumerate(pivot_cols):
            result[col] = b[i]

        return result


# ---------------------------------------------------------------------------
# Channel Simulation
# ---------------------------------------------------------------------------

def simulate_erasure_channel(
    packets: List[EncodedPacket],
    loss_rate: float,
    rng: Optional[np.random.Generator] = None,
) -> List[EncodedPacket]:
    """
    Simulate an i.i.d. packet erasure channel.

    Each packet is independently dropped with probability `loss_rate`.

    Args:
        packets   : Encoded packets to transmit.
        loss_rate : Packet loss probability in [0, 1].
        rng       : Optional RNG for reproducibility.

    Returns:
        Subset of packets that survived the channel.
    """
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.uniform(0, 1, size=len(packets)) > loss_rate
    return [pkt for pkt, keep in zip(packets, mask) if keep]


def bler_to_loss_rate(bler: float) -> float:
    """
    Convert wireless BLER to effective FC packet loss rate.
    Clamps to [0, 0.95] for numerical safety.
    """
    return float(np.clip(bler, 0.0, 0.95))


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

def _run_self_test(verbose: bool = True) -> dict:
    """
    Encode/decode at multiple loss rates and redundancy ratios.
    Validates Milestone 1.
    """
    results = {}
    k = 128
    fc = FountainCode(k=k, seed=42)

    test_configs = [
        # (redundancy_ratio, loss_rate, n_trials)
        (1.2, 0.00, 100),   # Minimal overhead, no loss
        (1.3, 0.00, 100),   # Default baseline, no loss
        (1.3, 0.05, 100),   # Default baseline, 5% loss
        (1.3, 0.10, 100),   # Default baseline, 10% loss
        (1.5, 0.10, 100),   # More overhead, 10% loss
        (1.5, 0.20, 100),   # More overhead, 20% loss
        (1.8, 0.30, 100),   # High overhead, 30% loss
        (2.0, 0.40,  50),   # Very high overhead, 40% loss
    ]

    for ratio, loss, n_trials in test_configs:
        successes = 0
        # Use a per-config RNG so results are reproducible
        trial_rng = np.random.default_rng(int(ratio * 1000) + int(loss * 1000))
        channel_rng = np.random.default_rng(int(ratio * 7777) + int(loss * 3333))

        for trial in range(n_trials):
            source = trial_rng.integers(0, 256, size=k, dtype=np.uint8)
            encoded = fc.encode(source, redundancy_ratio=ratio)
            received = simulate_erasure_channel(encoded, loss_rate=loss, rng=channel_rng)
            decoded = fc.decode(received, k=k)

            if decoded is not None and np.array_equal(decoded, source):
                successes += 1

        dsr = successes / n_trials
        key = f"ratio={ratio:.1f}_loss={loss:.0%}"
        results[key] = {
            "decode_success_rate": dsr,
            "n_trials": n_trials,
            "n_encoded": math.ceil(k * ratio),
            "ratio": ratio,
            "loss": loss,
        }
        if verbose:
            n_enc = math.ceil(k * ratio)
            tag = "PASS" if dsr >= 0.85 else "WARN"
            print(
                f"  ratio={ratio:.1f}  loss={loss:4.0%}  "
                f"n_encoded={n_enc:4d}  DSR={dsr:.1%}  "
                f"({tag})"
            )

    return results


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Milestone 1 -- Random Linear Fountain Code Self-Test")
    print("=" * 60)
    print(f"\nSource block size k=128")
    print(f"Code: Random Linear over GF(2), Gaussian elimination decoder")
    print(f"\nRunning encode/decode trials:\n")

    results = _run_self_test(verbose=True)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_pass = True
    for key, r in results.items():
        tag = "[PASS]" if r["decode_success_rate"] >= 0.85 else "[WARN]"
        print(f"  {key:35s}  DSR={r['decode_success_rate']:.1%}  {tag}")
        if r["loss"] == 0.0 and r["decode_success_rate"] < 0.99:
            all_pass = False

    print()
    if all_pass:
        print("[PASS] Milestone 1 PASSED -- Fountain Code is correct and ready.")
    else:
        print("[WARN] Milestone 1 has warnings -- review DSR at 0%% loss.")

    sys.exit(0 if all_pass else 1)
