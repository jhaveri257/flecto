import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n=========================================")
    print(f"Running: {script_path}")
    print(f"=========================================")
    try:
        # Run with the python launcher
        result = subprocess.run(
            ["py", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        print("Stdout:")
        print(result.stdout)
        if result.stderr:
            print("Stderr:")
            print(result.stderr)
        print(f"SUCCESS: {script_path} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {script_path} failed with exit code {e.returncode}.")
        print("Stdout:")
        print(e.stdout)
        print("Stderr:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"FAILED to start {script_path}: {e}")
        return False

def main():
    scripts = [
        "data_study/ablation.py",
        "data_study/plotting.py",
        "data_study/plotting_new.py",
        "RL/network.py",
        "RL/environment.py"
    ]
    
    results = {}
    for script in scripts:
        # Convert to proper path representation
        normalized_path = os.path.normpath(script)
        results[script] = run_script(normalized_path)
        
    print("\n=========================================")
    print("Execution Summary:")
    print("=========================================")
    for script, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{script:<30} : {status}")

if __name__ == "__main__":
    main()
