import time
import os
import sys

def mock_bandwidth(file_path, interval=0.1):
    print(f"Mocking bandwidth generation to {file_path} every {interval}s")
    # Ensure directory exists if needed
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    try:
        while True:
            # Write a mock bandwidth value (e.g., oscillating around 5.0 MB/s)
            with open(file_path, "a+") as f:
                bandwidth = 5.0 + (time.time() % 2)  # Some varying value between 5.0 and 7.0
                f.write(f"{int(time.time())} / {bandwidth} \n")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Mock bandwidth stopped.")

if __name__ == "__main__":
    from constants import bw_file_path
    mock_bandwidth(bw_file_path, 0.1)
