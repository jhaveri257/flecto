from typing import Iterable
import os, sys, time, zmq
from collections import deque
import subprocess
import json

def get_original_username():
    # Get the original username using the SUDO_USER environment variable
    return os.getenv("SUDO_USER") or os.getenv("USER")

user = get_original_username()

if user == None:
    directory_path = f"/ETTUS-data-collection/data_collection"
else:
    directory_path = f"/home/{user}/ETTUS-data-collection/data_collection"


directory = os.path.abspath(directory_path)
sys.path.append(directory)

from parsing import parse_L1, parse_MAC, parse_bw
from live_processing import clean_L1_list, clean_MAC_list

from ..packet_builder import QuicSentPacket
from .base import (
    K_MINIMUM_WINDOW,
    QuicCongestionControl,
    QuicRttMonitor,
    register_congestion_control,
)

from .rl import RL_agent

rl_agent = RL_agent()


def start_sniffing(path, interface, interval):
    global pid_sniffing
    if user == None:
        command = f"python3 /ETTUS-data-collection/data_collection/sniffing.py {path} {interface} {interval}"
    else:
        command = f"echo 3903 | sudo -S python3 /home/{user}/ETTUS-data-collection/data_collection/sniffing.py {path} {interface} {interval}"
    pid_sniffing = subprocess.Popen(command, shell=True)

def stop_sniffing():
    global pid_sniffing
    if pid_sniffing:
        pid_sniffing.terminate()
        pid_sniffing.wait()
        pid_sniffing = None

class RL(QuicCongestionControl):
    def __init__(self, *, max_datagram_size: int) -> None:
        super().__init__(max_datagram_size=max_datagram_size)
        self._max_datagram_size = max_datagram_size
        self._rtt_monitor = QuicRttMonitor()
        self.bytes_in_flight:int = 0
        self.reset()
        self.L1 = []
        self.MAC = []
        self.bw = 0
        if user == None:
            self.bw_path = "/ETTUS-data-collection/bandwidth.log"
        else:
            self.bw_path = f"/home/{user}/ETTUS-data-collection/bandwidth.log"
        start_sniffing(self.bw_path, "lo", 0.1)
        self.rtt = 0.02
        self.update_state()
        self.train = True
        self.state = []

    def reset(self, **kwargs):
        self.congestion_window = K_MINIMUM_WINDOW * self._max_datagram_size

    def on_packet_acked(self, *, now: float, packet: QuicSentPacket) -> None:
        self.bytes_in_flight -= packet.sent_bytes

        self.update_state()
        #print(self.state)

        #self.congestion_window, self.last_action = rl_agent.get_new_cw(self.get_state(), self.congestion_window)

        if self.train:
            self.congestion_window = rl_agent.get_new_cw(self.state, self.congestion_window)
            self.update_state()
            print(self.state)
            rl_agent.predict_and_learn(self.state, self.congestion_window)
        else:
            self.congestion_window = rl_agent.get_new_cw(self.state, self.congestion_window)
            #print(self.congestion_window)

    def on_packet_sent(self, *, packet: QuicSentPacket) -> None:
        self.bytes_in_flight += packet.sent_bytes

        self.update_state()

    def on_packets_expired(self, *, packets: Iterable[QuicSentPacket]) -> None:
        for packet in packets:
            self.bytes_in_flight -= packet.sent_bytes

        self.update_state()

    def on_packets_lost(self, *, now: float, packets: Iterable[QuicSentPacket]) -> None:
        for packet in packets:
            self.bytes_in_flight -= packet.sent_bytes
            
        self.update_state()
        stop_sniffing() # Not sure about this

    def read_and_parse_from_file(self, file_path, num_lines, parsing_function):
        lines = []
        try:
            with open(file_path, 'r') as file:
                last_lines = deque(file, maxlen=num_lines)

                for line in last_lines:
                    line = line.strip()
                    if line:
                        lines.append(parsing_function(line))
        except FileNotFoundError as e:
            print(f"Error: File not found: {file_path}")
                
        return lines

    def update_state(self):
        current_ts = int(time.time())
        #time.sleep(0.1)
        if user == None:
            path_L1 = "/low_level/nrL1_stats.log"
            path_MAC = "/low_level/nrMAC_stats.log"
        else:
            path_L1 = f"/home/{user}/openairinterface5g/cmake_targets/ran_build/build/nrL1_stats.log"
            path_MAC = f"/home/{user}/openairinterface5g/cmake_targets/ran_build/build/nrMAC_stats.log"

        try:
            lines_L1 = self.read_and_parse_from_file(path_L1, 50, parse_L1)
            #Search the index of the current timestamp in the L1 file, this should contain only one element
            found_indices_L1 = [index for index, item in enumerate(lines_L1) if current_ts in item[0]]
        except FileNotFoundError:
            print("L1 file not found!")
        try:
            lines_MAC = self.read_and_parse_from_file(path_MAC, 50, parse_MAC)
            #Search the index of the current timestamp in the MAC file, this should contain only one element
            found_indices_MAC = [index for index, item in enumerate(lines_MAC) if current_ts in item[0]]
        except:
            print("MAC file not found!")

        #print(self.read_and_parse_from_file(self.bw_path, 1, parse_bw))
        self.bw = self.read_and_parse_from_file(self.bw_path, 1, parse_bw)[0][0][1] #[([1719174479, 0.0], 'bw')]

        res_L1 = []
        res_MAC = []

        if found_indices_L1:
            for index in range(found_indices_L1[0], len(lines_L1)):
                item, flag = lines_L1[index]
                res_L1.append(item)
                if flag:
                    break
            res_L1 = clean_L1_list([item for sublist in res_L1 if sublist for item in sublist])
            res_L1 = [float(x) for x in res_L1]

        if res_L1:
            self.L1 = res_L1

        if found_indices_MAC:
            for index in range(found_indices_MAC[0], len(lines_MAC)):
                item, flag = lines_MAC[index]
                res_MAC.append(item)
                if flag:
                    break
            res_MAC = clean_MAC_list([item for sublist in res_MAC if sublist for item in sublist])
            res_MAC = [float(x) for x in res_MAC]

        if res_MAC:
            self.MAC = res_MAC

        #print(f"res_L1: {self.L1}")
        #print(f"res_MAC: {self.MAC}")
        #print(f"bw value: {self.bw}")
        #print(f"rtt value: {self.rtt}")
        #print(f"cw value: {self.congestion_window}")

        self.state = [self.bw] + [self.congestion_window] + self.L1[1:] + self.MAC[1:] + [self.rtt]

    def on_rtt_measurement(self, *, now: float, rtt: float) -> None:
        self.rtt = rtt
        # check whether we should exit slow start
        if self.ssthresh is None and self._rtt_monitor.is_rtt_increasing(
            rtt=rtt, now=now
        ):
            self.ssthresh = self.congestion_window


register_congestion_control("rl_based", RL)