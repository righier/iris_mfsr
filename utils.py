from multiprocessing import Pool, cpu_count
import torch
import gc

def p_umap(func, data, n_worker=1):
    with Pool(n_worker) as pool:
        out = list(tqdm(pool.imap_unordered(func, data), total=len(data)))
    return out

def to_gb(x):
  return x / 1024 / 1024 / 1024

def print_free_memory():
  if torch.cuda.is_available():
    t = to_gb(torch.cuda.get_device_properties(0).total_memory)
    r = to_gb(torch.cuda.memory_reserved(0))
    a = to_gb(torch.cuda.memory_allocated(0))
    f = t-a  # free inside reserved
    print("Total memory:", "{:.2f}".format(t), "GB")
    print("Reserved memory:", "{:.2f}".format(r), "GB")
    print("Allocated memory:", "{:.2f}".format(a), "GB")
    print("Free memory:", "{:.2f}".format(f), "GB")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

## returns cuda device if available
def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on:", "cpu" if device.type == "cpu" else torch.cuda.get_device_name(device))
    print_free_memory()
    return device

def get_cpu_device():
    return torch.device("cpu")