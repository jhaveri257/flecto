import paramiko

import multiprocessing  
import zmq
import json
import sys, os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from data_collection.parsing import parse_L1, parse_MAC, parse_quiche, parse_aioquic, parse_bw
from data_collection.constants import host, username, password, remote_file_path_L1, remote_file_path_MAC, remote_file_path_L4_aioquic, publisher_addr, host_ue, username_ue, password_ue

def live_read_remote_file(host, username, password, remote_file_path, parsing_function, publisher_addr,):
    # Create an SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.connect(publisher_addr)

    try:
        # Connect to the SSH server
        client.connect(hostname=host, username=username, password=password)

        # Open a SFTP session
        sftp = client.open_sftp()

        # Open the remote file
        remote_file = sftp.file(remote_file_path)
        remote_file.seek(0,2)

        block = []
        block_rtt = []
        block_cw = []
        block_bw = []
        ts = None
        current_ts = -1

        buffer = b''

        while True:
            new_data = remote_file.read(65536)
            buffer += new_data
            #line = remote_file.readline()
            #if not line:
            #    break  # No more data, exit loop
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                line = line.decode('utf-8')

                if line:
                    res, tag = parsing_function(line)
                    #block.append((res,tag))
                    #print(res, tag)
                    if tag == False:
                        block.append(res)
                    elif tag == True:
                        block.append(res)
                        message = json.dumps({"file_path": remote_file_path.split("/")[-1], "block": block})
                        publisher.send(message.encode())
                        block = []
                    elif tag == "rtt":
                        message = json.dumps({"file_path": "rtt", "block": res})
                        publisher.send(message.encode())
                    elif tag == "cw":
                        message = json.dumps({"file_path": "cw", "block": res})
                        publisher.send(message.encode())
                    elif tag == "bw":
                        message = json.dumps({"file_path": "bw", "block": res})
                        if res[1] > 1:
                            publisher.send(message.encode())
                    elif tag == "aioquic":
                        message = json.dumps({"file_path": "aioquic", "block": res})
                        publisher.send(message.encode())
                    elif tag == "skip":
                        pass

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("Closing connection...")
        if client:
            client.close()

if __name__ == "__main__":
    remote_file_path_bw = "/home/tst/bandwidth.log"
    process_L1 = multiprocessing.Process(target=live_read_remote_file, args=(host, username, password, remote_file_path_L1, parse_L1, publisher_addr))
    process_MAC = multiprocessing.Process(target=live_read_remote_file, args=(host, username, password, remote_file_path_MAC, parse_MAC, publisher_addr))
    #process_L4 = multiprocessing.Process(target=live_read_remote_file, args=(host, username, password, remote_file_path_L4, parse_quiche, publisher_addr))
    process_aioquic = multiprocessing.Process(target=live_read_remote_file, args=(host, username, password, remote_file_path_L4_aioquic, parse_aioquic, publisher_addr))
    process_bw = multiprocessing.Process(target=live_read_remote_file, args=(host_ue, username_ue, password_ue, remote_file_path_bw, parse_bw, publisher_addr))

    process_L1.start()
    process_MAC.start()
    #process_L4.start()
    process_aioquic.start()
    process_bw.start()