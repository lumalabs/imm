import argparse
import functools
import os
import socket
import subprocess
import sys
from typing import List

import argparse
import os 

import submitit
import copy
  
echo = lambda info: os_system(
    f'echo "[$(date "+%m-%d-%H:%M:%S")] ({os.path.basename(sys._getframe().f_back.f_code.co_filename)}, line{sys._getframe().f_back.f_lineno})=> {info}"')
os_system = functools.partial(subprocess.call, shell=True)
os_system_get_stdout = lambda cmd: subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')

 

def parse_args():
    parser = argparse.ArgumentParser("Submitit") 
  
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--port", default=28304, type=int, help="port")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request") 
    
    parser.add_argument("--partition", default="", type=str, help="Partition where to submit") 
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    return parser.parse_known_args()



class Trainer(object):
    def __init__(self, known_args, other_args):
        self.known_args, self.other_args = known_args, other_args
 
    def __call__(self):

        other_args: List[str] = copy.deepcopy(self.other_args)
        echo(f'[other_args received by launch.py]: {other_args}')
 
        if int(self.known_args.nodes) == 1:
            
            cmd = ( 
                f'torchrun --standalone --nnodes=1 --nproc_per_node={self.known_args.ngpus}  train.py'
                f' {" ".join(other_args)}'
            )
        else:
            
            cmd = ( 
                f'torchrun --nnodes={self.known_args.nodes} --nproc_per_node={self.known_args.ngpus} --rdzv-backend=c10d \
                        --rdzv-endpoint=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1):$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) train.py'
                f' {" ".join(other_args)}'
            ) 
        exit_code = subprocess.call(cmd, shell=True)
        sys.exit(exit_code)
        


def main():
    args, other_args = parse_args() 
    executor = submitit.AutoExecutor(folder=f'outputs/slurm_output', slurm_max_num_timeout=30)
    
    num_gpus_per_node = args.ngpus
    nodes = args.nodes 

    partition = args.partition
    kwargs = {}
    
    if args.comment:
        kwargs['slurm_comment'] = args.comment
        
    executor.update_parameters(
        mem_gb=2000,  
        tasks_per_node=1,  
        cpus_per_task=96, 
        nodes=nodes,
        timeout_min=24 * 60 * 30,
        # Below are cluster dependent parameters 
        slurm_partition=partition, 
        slurm_gres=f'gpu:{num_gpus_per_node}',
        slurm_signal_delay_s=120,  
        **kwargs
    )
    
 

    trainer = Trainer(args, other_args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()