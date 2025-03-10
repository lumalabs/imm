import os
import torch
from . import training_stats
import torch.distributed as dist
import datetime

# ----------------------------------------------------------------------------


def init(): 
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
        
    os.environ["NCCL_SOCKET_IFNAME"] = "enp" 
    os.environ["FI_EFA_SET_CUDA_SYNC_MEMOPS"] = "0"

    os.environ["NCCL_BUFFSIZE"] = "8388608"
    os.environ["NCCL_P2P_NET_CHUNKSIZE"] = "524288"
     
    
    os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '1200'
    os.environ['TORCH_NCCL_ENABLE_MONITORING'] = '0'
    
    backend = "gloo" if os.name == "nt" else "nccl"
    torch.distributed.init_process_group(backend=backend, init_method="env://",  timeout=datetime.timedelta(minutes=120),)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    sync_device = torch.device("cuda") if get_world_size() > 1 else None
    training_stats.init_multiprocessing(rank=get_rank(), sync_device=sync_device)


# ----------------------------------------------------------------------------


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


# ----------------------------------------------------------------------------


def get_world_size():
    return (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )


# ----------------------------------------------------------------------------


def should_stop():
    return False


# ----------------------------------------------------------------------------


def update_progress(cur, total):
    _ = cur, total


# ----------------------------------------------------------------------------


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


# ----------------------------------------------------------------------------

  
 
broadcast = dist.broadcast
new_group = dist.new_group
barrier = dist.barrier
all_gather = dist.all_gather
send = dist.send
recv = dist.recv