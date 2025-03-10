import os
import re
import json 

import pickle 
import functools 
import numpy as np

import torch
import dnnlib  

import torchvision.utils as vutils
import warnings
  
from omegaconf import OmegaConf
from torch_utils import misc 
import hydra

warnings.filterwarnings(
    "ignore", "Grad strides do not match bucket view strides"
)  # False warning printed by PyTorch 1.12.

 
 

# ----------------------------------------------------------------------------

def generator_fn(*args, name='pushforward_generator_fn', **kwargs):
    return globals()[name](*args, **kwargs)
 
 
 
 
@torch.no_grad()
def pushforward_generator_fn(net, latents, class_labels=None,  discretization=None, mid_nt=None,  num_steps=None,  cfg_scale=None, ):
    # Time step discretization.
    if discretization == 'uniform':
        t_steps = torch.linspace(net.T, net.eps, num_steps+1, dtype=torch.float64, device=latents.device) 
    elif discretization == 'edm':
        nt_min = net.get_log_nt(torch.as_tensor(net.eps, dtype=torch.float64)).exp().item()
        nt_max = net.get_log_nt(torch.as_tensor(net.T, dtype=torch.float64)).exp().item()
        rho = 7
        step_indices = torch.arange(num_steps+1, dtype=torch.float64, device=latents.device)
        nt_steps = (nt_max ** (1 / rho) + step_indices / (num_steps) * (nt_min ** (1 / rho) - nt_max ** (1 / rho))) ** rho
        t_steps = net.nt_to_t(nt_steps)
    else:
        if mid_nt is None:
            mid_nt = []
        mid_t = [net.nt_to_t(torch.as_tensor(nt)).item() for nt in mid_nt] 
        t_steps = torch.tensor(
            [net.T] + list(mid_t), dtype=torch.float64, device=latents.device
        )    
        # t_0 = T, t_N = 0
        t_steps = torch.cat([t_steps, torch.ones_like(t_steps[:1]) * net.eps])
     
    # Sampling steps
    x = latents.to(torch.float64)  
     
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                  
        x = net.cfg_forward(x, t_cur, t_next, class_labels=class_labels, cfg_scale=cfg_scale   ).to(
            torch.float64
        )   
         
        
    return x
 
@torch.no_grad()
def restart_generator_fn(net, latents, class_labels=None, discretization=None, mid_nt=None,  num_steps=None,  cfg_scale=None ):
    # Time step discretization.
    if discretization == 'uniform':
        t_steps = torch.linspace(net.T, net.eps, num_steps+1, dtype=torch.float64, device=latents.device)[:-1]
    elif discretization == 'edm':
        nt_min = net.get_log_nt(torch.as_tensor(net.eps, dtype=torch.float64)).exp().item()
        nt_max = net.get_log_nt(torch.as_tensor(net.T, dtype=torch.float64)).exp().item()
        rho = 7
        step_indices = torch.arange(num_steps+1, dtype=torch.float64, device=latents.device)
        nt_steps = (nt_max ** (1 / rho) + step_indices / (num_steps) * (nt_min ** (1 / rho) - nt_max ** (1 / rho))) ** rho
        t_steps = net.nt_to_t(nt_steps)[:-1]
    else:
        if mid_nt is None:
            mid_nt = []
        mid_t = [net.nt_to_t(torch.as_tensor(nt)).item() for nt in mid_nt] 
        t_steps = torch.tensor(
            [net.T] + list(mid_t), dtype=torch.float64, device=latents.device
        )     
    # Sampling steps
    x = latents.to(torch.float64)  
     
    for i, t_cur in enumerate(t_steps):
         
               
        x = net.cfg_forward(x, t_cur, torch.ones_like(t_cur) * net.eps, class_labels=class_labels,  cfg_scale=cfg_scale  ).to(
            torch.float64
        )    
            
        if i < len(t_steps) - 1:
            x, _ = net.add_noise(x, t_steps[i+1])
            
    return x
 
 
 
# ----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs")
def main(cfg):

    device = torch.device("cuda")
    config = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True)) 
  
    # Random seed.
    if config.eval.seed is None:

        seed = torch.randint(1 << 31, size=[], device=device)
        torch.distributed.broadcast(seed, src=0)
        config.eval.seed = int(seed)

    # Checkpoint to evaluate.
    resume_pkl = cfg.eval.resume 
    cudnn_benchmark = config.eval.cudnn_benchmark  
    seed = config.eval.seed
    encoder_kwargs = config.encoder
    
    batch_size = config.eval.batch_size
    sample_kwargs_dict = config.get('sampling', {})
    # Initialize.
    np.random.seed(seed % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
  
    print('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs) 
 
    # Construct network.
    print("Constructing network...") 

    interface_kwargs = dict(
        img_resolution=config.resolution,
        img_channels=config.channels,
        label_dim=config.label_dim,
    )
    if config.get('network', None) is not None:
        network_kwargs = config.network
        net = dnnlib.util.construct_class_by_name(
            **network_kwargs, **interface_kwargs
        )  # subclass of torch.nn.Module
        net.eval().requires_grad_(False).to(device)
  
    # Resume training from previous snapshot.   
    with dnnlib.util.open_url(resume_pkl, verbose=True) as f:
        data = pickle.load(f) 
    
    if config.get('network', None) is not None: 
        misc.copy_params_and_buffers(
            src_module=data['ema'], dst_module=net, require_all=True
        ) 
    else:
        net = data['ema'].eval().requires_grad_(False).to(device)
                
       
    grid_z = net.get_init_noise(
        [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
        device,
    )  
    if net.label_dim > 0: 
        labels = torch.randint(0, net.label_dim, (batch_size,), device=device)
        grid_c = torch.nn.functional.one_hot(labels,  num_classes=net.label_dim)
    else:
        grid_c = None
 
    # Few-step Evaluation. 
    generator_fn_dict = {k: functools.partial(generator_fn, **sample_kwargs) for k, sample_kwargs in sample_kwargs_dict.items()}
    print("Sample images...") 
    res = {}
    for key, gen_fn in generator_fn_dict.items():
        images = gen_fn(net, grid_z, grid_c)   
        images = encoder.decode(images.to(device) ).detach().cpu() 
        
        vutils.save_image(
            images / 255.,
            os.path.join(f"{key}_samples.png"),
            nrow=int(np.sqrt(images.shape[0])),
            normalize=False,
        )
        
        res[key] = images  
    
    print('done.')

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
