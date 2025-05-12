import subprocess
import time

from wandb_sync import sync_wandb
    
def loop(start_time, functions, min_time=200, t=4): # in seconds
    running = True
    while running or time.time() - start_time < min_time:
        for f in functions:
            f()
        running = len(subprocess.check_output("qstat")) > 0
        time.sleep(t)
        
if __name__ == "__main__":
    start_time = time.time()
    loop(start_time, [sync_wandb])