# pyrefly: ignore [missing-import]
import zmq
import json
from collections import defaultdict

import csv

import queue
from queue import Queue
from threading import Thread
import time

import sys, os, inspect, socket

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from data_collection.constants import print_to_file, file_path_final, subscriber_addr, column_headers, wanted_entries, SERVER_HOST, SERVER_PORT
from data_collection.utils import human_readable_number

def send_message():
    print("Sending message")
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    client_socket.connect((SERVER_HOST, SERVER_PORT))

    # Send a message to the server
    message = "NO LOGS"
    client_socket.send(message.encode('utf-8'))

    # Close the connection
    #client_socket.close()

def clean_MAC_list(l):
    return [x for i, x in enumerate(l) if i not in [1, 2]]

def clean_L1_list(l):
    return [l[0]] + [x for i, x in enumerate(l) if i not in [0, 9, 13, 15]]

def merge_by_sec(blocks):
    merged_data = defaultdict(dict)
    tags = ['nrL1_stats.log', 'nrMAC_stats.log']

    #Just copy if tag in tags, else create a list for each tag-timestamp pair and append all the data there
    for tag, line_data in blocks:
        timestamp, *data_values = line_data
        if tag in tags:
            merged_data[tag][timestamp] = data_values
        else:
            if tag not in merged_data:
                merged_data[tag] = {}
            if timestamp not in merged_data[tag]:
                merged_data[tag][timestamp] = []
            merged_data[tag][timestamp].append(data_values)

    tag_to_check = "aioquic"

    tag_found = any(tag == tag_to_check for tag in merged_data.keys())

    if tag_found:
        cw_data = []
        rtt_data = []
        modified_data = {'cw': {}, 'rtt': {}}
        for tag, timestamp_data in merged_data.items():
            # Check if the tag is 'aioquic'
            if tag == 'aioquic':
                for timestamp, value_list in timestamp_data.items():
                    # Extract the elements from the list
                    element_cw = [[sublist[0]] for sublist in value_list]
                    elements_rtt = [sublist[1:] for sublist in value_list]

                    # Add the values to the modified dictionary
                    modified_data['cw'][timestamp] = element_cw
                    modified_data['rtt'][timestamp] = elements_rtt
        
        # Update merged_data with modified_data
        merged_data.update(modified_data)

    selected_tags = ['bw', 'cw', 'rtt']

    for tag, timestamp_data in merged_data.items():
        if tag in selected_tags:
            for timestamp, value_list in timestamp_data.items():
                list_of_lists_int = [[int(value) for value in sublist] for sublist in value_list]
                t = zip(*list_of_lists_int)
                avg = [sum(x) // len(x) for x in t]
                merged_data[tag][timestamp] = avg

    merged_data = {k: merged_data[k] for k in sorted(merged_data)}

    merged_timestamp_values = {}

    if "aioquic" in merged_data.keys():
        del merged_data["aioquic"]        

    tags_to_check = ['nrL1_stats.log', 'nrMAC_stats.log', 'bw', 'cw', 'rtt']

    if all(tag in merged_data for tag in tags_to_check):
        for timestamp, tag_values in merged_data[next(iter(merged_data))].items():
            if all(timestamp in merged_data[tag] for tag in merged_data):
                merged_values = []
                for tag in merged_data:
                    merged_values.extend(merged_data[tag][timestamp])
                merged_timestamp_values[timestamp] = merged_values

    # Convert merged_timestamp_values into a list of lists
    merged_list = [[timestamp] + values for timestamp, values in merged_timestamp_values.items()]

    return merged_list

def count_lines_in_csv(filename):
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        line_count = len(list(csv_reader))
    return line_count

def has_header(filename):
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        first_row = next(reader, None)  # Read the first row
        if first_row:
            # Check if the first row contains non-empty values
            return any(field.strip() for field in first_row)
        return False  # File is empty

def process_messages(subscriber_addr, file_path_final, wanted_entries, print_to_file=True):
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.bind(subscriber_addr)
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')

    # Define a buffer queue
    buffer_queue = Queue(maxsize=100000)  # Maximum buffer size

    def process_buffer():
        L1_time = time.time()
        blocks = []
        while True:
            # Get a message from the buffer queue

            try:
                msg = buffer_queue.get(timeout=1)
                buffer_queue.task_done()
                # Process the message
                data = json.loads(msg.decode())
                file_path = data['file_path']
                if file_path == "rtt" or file_path == "cw" or file_path == "bw" or file_path == "aioquic":
                    merged_list = data['block']
                elif file_path == "nrL1_stats.log":
                    merged_list = clean_L1_list([item for sublist in data['block'] if sublist for item in sublist])
                    L1_time = time.time()
                else:
                    merged_list = clean_MAC_list([item for sublist in data['block'] if sublist for item in sublist])
                blocks.append((file_path, merged_list))
                
                if len(blocks) % 1000 == 0:
                    blocks_merged_len = len(merge_by_sec(blocks))
                    print(f"Number of blocks so far: {human_readable_number(len(blocks))}")
                    print(f"Number of entries so far in memory: {blocks_merged_len}")
                
                if len(blocks) > 10_000:
                    merged_blocks = merge_by_sec(blocks)
                    with open(file_path_final, "a") as f:
                        writer = csv.writer(f)
                        writer.writerows(merged_blocks)
                    cnt = count_lines_in_csv(file_path_final)
                    print(f"Number of entries in file so far: {cnt - 1}")
                    if cnt - 1 > wanted_entries:
                        print(f"Final entries: {cnt - 1}")
                        break
                    blocks = []

            except queue.Empty:
                print(f"Buffer queue is empty for {time.time() - L1_time} seconds")
                if time.time() - L1_time > 10:
                    try:
                        send_message()
                    except Exception as e:
                        print(f"[warn] Could not reach UE server (expected in local mode): {e}")
                    time.sleep(15)

            except json.decoder.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Received message: {msg}")
                  # Signal that message processing is complete

    # Start processing buffer in a separate thread
    buffer_thread = Thread(target=process_buffer)
    buffer_thread.daemon = True
    buffer_thread.start()

    try:
        while True:
            # Receive messages from subscriber and put them into buffer queue
            msg = subscriber.recv()
            buffer_queue.put(msg)

    except KeyboardInterrupt:
        print("Processing script terminated by user")
    finally:
        # Wait for buffer processing to complete before closing sockets and terminating context
        buffer_queue.join()
        print("Goodbye...")
        subscriber.close()
        context.term()

if __name__ == "__main__":
    if print_to_file:
        if os.path.exists(file_path_final) and has_header(file_path_final):
            pass
        else:    
            with open(file_path_final, "w+") as f:
                writer = csv.writer(f)
                writer.writerow(column_headers)
    process_messages(subscriber_addr, file_path_final, wanted_entries, print_to_file)