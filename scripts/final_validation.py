import subprocess
import time
import requests
import sys

def main():
    print("Starting Flask Backend...")
    server = subprocess.Popen([sys.executable, "app/app.py"])
    
    # wait for server to start
    time.sleep(10)
    
    print("Running API test...")
    try:
        subprocess.run([sys.executable, "scripts/test_api.py"], check=True)
    except Exception as e:
        print(f"Error running test_api: {e}")
    
    print("Killing server...")
    server.terminate()
    print("Final validation complete.")

if __name__ == "__main__":
    main()
