import subprocess, os

def parse(x):
    return int(x.split(", ")[1].split(" MiB")[0])

cap = 50 # MiB

def gpu_memory_usage():
    command = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv"]
    output = subprocess.check_output(command).decode("utf-8").split("\n")[1:-1]
    print(f"Output: {output}")
    available = [parse(x) < cap for x in output]
    indices = [int(x.split(", ")[0]) for x in output]   
    print(f"Available GPUs: {available}")
    # reindex to make sure that the indices are in order
    aid = [x for x in indices if available[indices.index(x)]]
    
    return aid, len(available)

# get a gpu in 0-4 and a gpu in 5-9
def set_gpu(ngpu):

    aid, n_host = gpu_memory_usage()
    
    if n_host == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        return
    
    g1 = [x for x in aid if x < 5]
    g2 = [x for x in aid if x >= 5]
    print(g1, g2)
    if ngpu == 1:
        g1.extend(g2)
        assert len(g1) > 0, "No GPU available"
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{g1[0]}"
    else:
        assert len(g1) > 0 and len(g2) > 0 , "No paired across system GPUs available"
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{g1[0]},{g2[0]}"
