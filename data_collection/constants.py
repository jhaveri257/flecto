import os

def get_original_username():
    # Get the original username using the SUDO_USER environment variable
    return os.getenv("SUDO_USER") or os.getenv("USER")

host = '10.172.13.150'
username = 'core'
password = 'core'
remote_file_path_L1 = '/home/core/openairinterface5g/cmake_targets/ran_build/build/nrL1_stats.log'
remote_file_path_MAC = '/home/core/openairinterface5g/cmake_targets/ran_build/build/nrMAC_stats.log'
remote_file_path_L4 = '/home/core/chromium/src/logging/nrL4_stats.log'
remote_file_path_L4_aioquic = '/home/core/ETTUS-data-collection/aioquic/logs/aioquic.log'
publisher_addr = "tcp://127.0.0.1:5555"
_default_file_path = "/home/cristiano/ETTUS-data-collection/stats_cubic_aioquic.csv"
_local_fallback = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stats_local_output.csv"))
# Use local fallback on Windows or when the default Linux path is not accessible
file_path_final = _default_file_path if os.path.isdir(os.path.dirname(_default_file_path)) else _local_fallback
print_to_file = True
subscriber_addr = "tcp://127.0.0.1:5555"
column_headers = ["timestamp", "bw_avg", "cw_avg", "blacklisted_PRBs", "total_PRBs", "max_IO", "max_IO_par", "min_IO", "min_IO_par", "avg_IO", "PRACH_IO", "current_QM_DL", "current_RI_DL",
                  "total_bytes_TX", "ulsch_power", "ulsch_noise_power", "sync_pos", "round_trials_0", "round_trials_0_par", "round_trials_1", "round_trials_1_par", "round_trials_2", "round_trials_2_par",
                  "round_trials_3", "DTX", "current_QM_UL", "current_RI_UL", "total_bytes_RX", "total_bytes_scheduled", "PH", "PCMAX", "RSRP", "meas", "UL_RI", "TPMI",
                  "dlsch_rounds_0", "dlsch_rounds_1", "dlsch_rounds_2", "dlsch_rounds_3", "dlsch_errors", "pucch0_DTX", "BLER_DL", "MCS_DL", "dlsch_total_bytes",
                  "ulsch_rounds_0", "ulsch_rounds_1", "ulsch_rounds_2", "ulsch_rounds_3", "ulsch_DTX", "ulsch_errors", "BLER_UL", "MCS_UL", "ulsch_total_bytes_scheduled",
                  "ulsch_total_bytes_received", "smoothed_rtt", "rtt_mean_deviation"]
command_quiche = "/home/tst/chromium/src/out/Debug/quic_client --host=192.168.70.135 --port=6121 --interface_name=oaitun_ue1 --disable_certificate_verification https://www.example.org/quic-data/www.example.org/small_file.html"
command_aioquic = "python /home/tst/ETTUS-data-collection/aioquic/examples/http3_client.py --ca-certs /home/tst/ETTUS-data-collection/aioquic/tests/pycacert.pem --insecure --congestion-control-algorithm reno https://192.168.70.135:6121"
wanted_entries = 50_000
current_user = get_original_username() or "default_user"
#print(current_user)
_default_bw = f"/home/{current_user}/bandwidth.log"
_local_bw = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bandwidth.log"))
bw_file_path = _default_bw if os.path.isdir(f"/home/{current_user}") else _local_bw
interface = "oaitun_ue1"
port = 4433
host_ue = "10.172.13.21"
username_ue = "tst"
password_ue = "123"

SERVER_HOST = '10.172.13.21'  # Change this to the server's IP address if necessary
SERVER_PORT = 12345