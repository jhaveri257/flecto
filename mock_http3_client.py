import time
import sys

def main():
    print("Mock HTTP3 client started.", flush=True)
    print("Pretending to connect to QUIC server...", flush=True)
    time.sleep(1) # Simulate some delay
    print("Data transfer complete.", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()
