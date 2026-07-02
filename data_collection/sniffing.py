from scapy.all import sniff, UDP
import time
import sys, os, inspect, subprocess
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from data_collection.constants import bw_file_path, interface, port

# Function to calculate bandwidth
def calculate_bandwidth(start_time, prev_bytes, current_bytes):
    elapsed_time = time.time() - start_time
    # Convert bytes to megabytes
    bytes_per_second = (current_bytes - prev_bytes) / elapsed_time
    megabytes_per_second = bytes_per_second / (1024 * 1024)  # Convert bytes to kilobytes
    return megabytes_per_second

def process_packet(packet, port, bandwidth_data):
    if UDP in packet:
        udp_packet = packet[UDP]
        src_port = udp_packet.sport
        dst_port = udp_packet.dport

        if src_port == port or dst_port == port:
            current_bytes = len(packet)
            bandwidth_data[0] += current_bytes

# Main function
def get_bandwidth_every_sec(file_path, interface, interval = 0.1):
    interval = float(interval)

    f = open(file_path, "a+")
    # Specify the UDP port you want to monitor

    # Start packet sniffing
    #print("Monitoring bandwidth (megabytes per second) on port", port)
    while True:
        start_time = time.time()
        bandwidth_data = [0]
        
        sniff(iface=interface,filter="udp and port {}".format(port), prn=lambda packet: process_packet(packet, port, bandwidth_data), timeout=interval)

        current_time = time.time()
        elapsed_time = current_time - start_time
        bandwidth = calculate_bandwidth(start_time, 0, bandwidth_data[0])
        f.write(f"{int(time.time())} / {bandwidth} \n")
        f.flush()

        sleep_time = max(interval - elapsed_time, 0)  # Ensure sleep_time is non-negative
        time.sleep(sleep_time)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        interface = sys.argv[2]
        interval = sys.argv[3]
    else:
        path = bw_file_path
        interface = interface
        interval = 1
    get_bandwidth_every_sec(path, interface, interval)