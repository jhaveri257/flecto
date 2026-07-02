import subprocess
import os
import sys
import inspect
import psutil
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from data_collection.constants import command_quiche, command_aioquic, bw_file_path
from time import time, sleep
import threading
import random
import socket, select

pid = None
pid_request = None
pid_sniffing = None

def run_command(command):
    global pid_request
    # Redirect stdout and stderr to /dev/null
    try:
        with open(os.devnull, 'w') as devnull:
            subprocess.run(command, shell=True, stdout=devnull, stderr=devnull, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"Command timed out")

def start_ue():
    print("Restarting UE...")
    password = "123"
    cmd = f"echo {password} | sudo -S /home/tst/openairinterface5g/cmake_targets/ran_build/build/nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --ue-fo-compensation --sa -E --uicc0.imsi 001010000000001"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return process.pid

def ping(ip_address):
    print(f"Pinging {ip_address}...")
    # Run the ping command
    process = subprocess.Popen(['ping', '-c', '1', ip_address], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the command to finish and get the output
    stdout, stderr = process.communicate()

    # Check the return code to determine if the ping was successful
    return process.returncode == 0

def check_route():
    password = "123"
    cmd = f"echo {password} | sudo -S ip route list | grep 70."
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if stdout.decode().strip() == "192.168.70.128/26 dev oaitun_ue1 scope link":
        print("Route correctly added!")

    if stderr:
        print(f"ip route list output error: {stderr.decode()}")

def add_route_and_check():
    password = "123"
    print("Ping failed, adding route...")
    cmd = f"echo {password} | sudo -S ip route add 192.168.70.128/26 dev oaitun_ue1"
    run_command(cmd)
    check_route()

def restart_ue():
    global pid_request
    global pid
    if pid is None:
        print("Not killing PID")
    else:
        print(f"Killing pid: {pid}")
        try:
            process = psutil.Process(pid)
            print(f"Killing process with PID: {pid}")
            process.terminate()  # Send termination signal
            #time.sleep(1)  # Wait for the process to terminate (optional)
            print(f"Process with PID {pid} terminated successfully.")
        except psutil.NoSuchProcess:
            print(f"No process found with PID {pid}.")

    pid = start_ue()
    print("Started UE!")
    sleep(10)
    ip = "192.168.70.135"
    password = "123"
    if not ping(ip):
        add_route_and_check()
        if ping(ip):
            print("Ping successfull")
    return pid

def start_sniffing():
    global pid_sniffing
    print("Sniffing like Maradona...")
    command = f"echo 123 | sudo -S python sniffing.py"
    process = subprocess.Popen(command, shell=True)
    pid_sniffing = process.pid

def thread():
    global pid
    global pid_sniffing
    HOST = '0.0.0.0'
    PORT = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    clients = []

    while True:
        # Use select to wait for incoming data or connections
        ready_to_read, _, _ = select.select([server_socket] + clients, [], [])

        for sock in ready_to_read:
            if sock is server_socket:
                # Accept new connection
                client_socket, client_address = server_socket.accept()
                clients.append(client_socket)
                print(f"Connection established from {client_address}")
            else:
                # Handle incoming data from client
                data = sock.recv(1024).decode('utf-8')
                if not data:
                    # Client disconnected
                    print(f"Client {sock.getpeername()} disconnected")
                    sock.close()
                    clients.remove(sock)
                else:
                    print(f"Received message from {sock.getpeername()}: {data}")
                    if data == "NO LOGS":
                        print("Received NO LOGS message")
                        pid = restart_ue()
                        if pid_sniffing is not None:
                            try:
                                process = psutil.Process(pid_sniffing)
                                print(f"Killing process with PID: {pid_sniffing}")
                                process.terminate()  # Send termination signal
                                #time.sleep(1)  # Wait for the process to terminate (optional)
                                print(f"Process with PID {pid_sniffing} terminated successfully.")
                            except psutil.NoSuchProcess:
                                print(f"No process found with PID {pid_sniffing}.")
                        start_sniffing()

    # Close the server socket (not reached in this example)
    server_socket.close()

if __name__ == "__main__":
    arg = sys.argv[1]
    cnt = 1
    restart_ue()
    print(f"Starting command: {cnt}")

    start_sniffing()
    check_thread = threading.Thread(target=thread, args=())
    check_thread.start()
    
    while True:
        start = time()
        cnt += 1
        if arg == "aioquic":
            size = random.randint(500_000, 500_000_000)
            command = command_aioquic + f"/{size}"
        else:
            command = command_quiche
        run_command(command)
        time_taken = time() - start
        print(f"Starting command: {cnt} Time taken: {time_taken} seconds Size: {size}")
        #if time_taken < 10:
        #    restart_ue()

    process_sniffing.terminate()