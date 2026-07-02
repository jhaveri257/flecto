import subprocess, sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
client_py = os.path.join(script_dir, "mock_http3_client.py")
cert_pem = os.path.join(script_dir, "aioquic", "tests", "pycacert.pem")
command = f"{sys.executable} {client_py}"

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    
    # Read stdout and stderr in real time
    for stdout_line in iter(process.stdout.readline, ""):
        print(stdout_line.strip(), flush=True)
    for stderr_line in iter(process.stderr.readline, ""):
        print(stderr_line.strip(), file=sys.stderr, flush=True)

    process.stdout.close()
    process.stderr.close()
    #print("Waiting...")
    return_code = process.wait()
    return return_code

counter = 1

while True:
    #print(f"Running command: {command}")
    print(f"Running for: {counter}")
    counter += 1
    run_command(command)