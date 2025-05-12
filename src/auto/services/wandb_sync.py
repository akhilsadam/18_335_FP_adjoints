import os

def sync_wandb():
    os.system('wandb sync --include-offline --sync-all')