"""
mock_publisher.py  –  Simulate the live data-collection pipeline locally.

Reads rows from stats_cubic_aioquic.csv, splits each row back into the
four message streams that live_processing.py expects, and publishes them
via ZeroMQ.

Message formats (matching file_live.py exactly):
  • nrL1_stats.log  → {"file_path": "nrL1_stats.log", "block": [[v], [v], ...]}
    flat after clean_L1_list: [ts, L1_col_0, L1_col_1, ...]
  • nrMAC_stats.log → {"file_path": "nrMAC_stats.log", "block": [[v], [v], ...]}
    flat after clean_MAC_list: [ts, MAC_col_0, MAC_col_1, ...]
  • aioquic         → {"file_path": "aioquic", "block": [ts, cwnd, rtt, rtt_var]}
  • bw              → {"file_path": "bw",      "block": [ts, bw_kbps]}

Usage:
  # Terminal 1 (start subscriber first):
  py data_collection/live_processing.py

  # Terminal 2 (then start publisher):
  py data_collection/mock_publisher.py [--csv path/to/file.csv] [--rate 20]
"""

import json
import time
import argparse
import os
import sys
import pandas as pd
import zmq

# ── Column mapping ────────────────────────────────────────────────────────────
# After clean_L1_list, the L1 flat list is:
#   [ts, idx0..8, idx10..12, idx14, idx16..]   (removes original indices 9,13,15)
#
# The raw block sent by file_live.py for L1 is a nested list of lists.
# clean_L1_list flattens it first: [item for sublist in block if sublist for item in sublist]
# then removes indices 9,13,15 from that flattened list.
#
# We reconstruct a block where every element is [value], so flattening yields
# [ts, col1, col2, ...] and then clean_L1_list removes positions 9, 13, 15.
#
# L1 flat list BEFORE cleaning (positions 0-based):
#  0: timestamp
#  1: blacklisted_PRBs
#  2: total_PRBs
#  3: max_IO
#  4: max_IO_par
#  5: min_IO
#  6: min_IO_par
#  7: avg_IO
#  8: PRACH_IO
#  9: current_QM_DL      ← removed by clean_L1_list
# 10: current_RI_DL
# 11: total_bytes_TX
# 12: ulsch_power
# 13: ulsch_noise_power   ← removed by clean_L1_list
# 14: sync_pos
# 15: round_trials_0      ← removed by clean_L1_list
# 16: round_trials_0_par
# 17: round_trials_1
# 18: round_trials_1_par
# 19: round_trials_2
# 20: round_trials_2_par
# 21: round_trials_3
# 22: DTX  (this gets sent as "True" flag — it's the last element that triggers the block)
#
# After clean_L1_list removes 9,13,15 we get indices 0-8,10-12,14,16-22 → 20 values + ts

L1_COLS_ORDERED = [
    "blacklisted_PRBs", "total_PRBs", "max_IO", "max_IO_par", "min_IO",
    "min_IO_par", "avg_IO", "PRACH_IO", "current_QM_DL",   # 9 cols (idx 1-9)
    "current_RI_DL", "total_bytes_TX", "ulsch_power",        # idx 10-12
    "ulsch_noise_power",                                      # idx 13
    "sync_pos",                                               # idx 14
    "round_trials_0",                                         # idx 15
    "round_trials_0_par", "round_trials_1", "round_trials_1_par",
    "round_trials_2", "round_trials_2_par", "round_trials_3", "DTX",
]

# MAC block before cleaning:  [ts, PH, PCMAX, RSRP, ...]
# clean_MAC_list removes indices 1 and 2 (PH, PCMAX), keeping everything else.
MAC_COLS_ORDERED = [
    "PH", "PCMAX",                                            # idx 1,2 — stripped
    "RSRP", "meas", "UL_RI", "TPMI",
    "dlsch_rounds_0", "dlsch_rounds_1", "dlsch_rounds_2", "dlsch_rounds_3",
    "dlsch_errors", "pucch0_DTX", "BLER_DL", "MCS_DL", "dlsch_total_bytes",
    "ulsch_rounds_0", "ulsch_rounds_1", "ulsch_rounds_2", "ulsch_rounds_3",
    "ulsch_DTX", "ulsch_errors", "BLER_UL", "MCS_UL",
    "ulsch_total_bytes_scheduled", "ulsch_total_bytes_received",
]

def val(row, col):
    """Safely get a numeric value from a row."""
    v = row.get(col, 0)
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0


def publish_row(publisher, row: dict):
    ts = val(row, "timestamp")

    # ── L1 block ──────────────────────────────────────────────────────────────
    # Each element wrapped in its own list so flattening by clean_L1_list works
    l1_flat = [ts] + [val(row, c) for c in L1_COLS_ORDERED]
    l1_block = [[v] for v in l1_flat]
    publisher.send(json.dumps({"file_path": "nrL1_stats.log", "block": l1_block}).encode())

    # ── MAC block ─────────────────────────────────────────────────────────────
    mac_flat = [ts] + [val(row, c) for c in MAC_COLS_ORDERED]
    mac_block = [[v] for v in mac_flat]
    publisher.send(json.dumps({"file_path": "nrMAC_stats.log", "block": mac_block}).encode())

    # ── aioquic block ─────────────────────────────────────────────────────────
    # Format: [ts, cwnd, smoothed_rtt, rtt_variance]
    aioquic_data = [ts, val(row, "cw_avg"), val(row, "smoothed_rtt"), val(row, "rtt_mean_deviation")]
    publisher.send(json.dumps({"file_path": "aioquic", "block": aioquic_data}).encode())

    # ── bw block ─────────────────────────────────────────────────────────────
    # Format: [ts, bw_kbps]
    bw_data = [ts, val(row, "bw_avg")]
    publisher.send(json.dumps({"file_path": "bw", "block": bw_data}).encode())


def main():
    parser = argparse.ArgumentParser(description="Mock ZMQ publisher for ETTUS data collection")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.normpath(os.path.join(script_dir, "..", "stats_cubic_aioquic.csv"))
    parser.add_argument("--csv", default=default_csv, help="Path to source CSV file")
    parser.add_argument("--rate", type=float, default=20.0, help="Rows to publish per second")
    parser.add_argument("--addr", default="tcp://127.0.0.1:5555", help="ZMQ address to connect to")
    parser.add_argument("--loop", action="store_true", help="Loop the CSV infinitely")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"[mock_publisher] Loading: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"[mock_publisher] Loaded {len(df)} rows.")

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.connect(args.addr)
    print(f"[mock_publisher] Connected to {args.addr} (subscriber must bind there)")
    # Give SUB socket time to set up subscription filter
    time.sleep(1.5)

    delay = 1.0 / args.rate
    rows_sent = 0
    try:
        while True:
            for _, row in df.iterrows():
                publish_row(publisher, row.to_dict())
                rows_sent += 1
                if rows_sent % 200 == 0:
                    print(f"[mock_publisher] Published {rows_sent} rows…")
                time.sleep(delay)
            if not args.loop:
                break
            print(f"[mock_publisher] Finished one pass ({rows_sent} rows). Looping…")
    except KeyboardInterrupt:
        print(f"\n[mock_publisher] Stopped after {rows_sent} rows.")
    finally:
        publisher.close()
        context.term()
    print(f"[mock_publisher] Done. Total rows published: {rows_sent}")


if __name__ == "__main__":
    main()
