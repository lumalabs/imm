# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch_utils import persistence
from training.unets import *  
from training.dit import *  
 

@persistence.persistent_class
class IMMPrecond(torch.nn.Module):

    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        mixed_precision=None,   
        noise_schedule="fm",   
        model_type="SongUNet",   
        sigma_data=0.5, 
        f_type="euler_fm",
        T=0.994,
        eps=0.,  
        temb_type='identity', 
        time_scale=1000.,  
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
 

        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.label_dim = label_dim
        self.use_mixed_precision = mixed_precision is not None
        if mixed_precision == 'bf16':
            self.mixed_precision = torch.bfloat16
        elif mixed_precision == 'fp16':
            self.mixed_precision = torch.float16 
        elif mixed_precision is None:
            self.mixed_precision = torch.float32
        else:
           raise ValueError(f"Unknown mixed_precision: {mixed_precision}")
            
            
        self.noise_schedule = noise_schedule
 
        self.T = T
        self.eps = eps

        self.sigma_data = sigma_data

        self.f_type = f_type
 
        self.nt_low = self.get_log_nt(torch.tensor(self.eps, dtype=torch.float64)).exp().numpy().item()
        self.nt_high = self.get_log_nt(torch.tensor(self.T, dtype=torch.float64)).exp().numpy().item()
         
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            img_channels=img_channels,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )
        print('# Mparams:', sum(p.numel() for p in self.model.parameters()) / 1000000)
        
        self.time_scale = time_scale 
         
         
        self.temb_type = temb_type
        
        if self.f_type == 'euler_fm':
            assert self.noise_schedule == 'fm'
          

    def get_logsnr(self, t):
        dtype = t.dtype
        t = t.to(torch.float64)
        if self.noise_schedule == "vp_cosine":
            logsnr = -2 * torch.log(torch.tan(t * torch.pi * 0.5))
 
        elif self.noise_schedule == "fm":
            logsnr = 2 * ((1 - t).log() - t.log())
            
        logsnr = logsnr.to(dtype)
        return logsnr
    
    def get_log_nt(self, t):
        logsnr_t = self.get_logsnr(t)

        return -0.5 * logsnr_t
    
    def get_alpha_sigma(self, t): 
        if self.noise_schedule == 'fm':
            alpha_t = (1 - t)
            sigma_t = t
        elif self.noise_schedule == 'vp_cosine': 
            alpha_t = torch.cos(t * torch.pi * 0.5)
            sigma_t = torch.sin(t * torch.pi * 0.5)
            
        return alpha_t, sigma_t 

    def add_noise(self, y, t,   noise=None):

        if noise is None:
            noise = torch.randn_like(y) * self.sigma_data

        alpha_t, sigma_t = self.get_alpha_sigma(t)
         
        return alpha_t * y + sigma_t * noise, noise 

    def ddim(self, yt, y, t, s, noise=None):
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        alpha_s, sigma_s = self.get_alpha_sigma(s)
        

        if noise is None: 
            ys = (alpha_s -   alpha_t * sigma_s / sigma_t) * y + sigma_s / sigma_t * yt
        else:
            ys = alpha_s * y + sigma_s * noise
        return ys
  
   

    def simple_edm_sample_function(self, yt, y, t, s ):
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        alpha_s, sigma_s = self.get_alpha_sigma(s)
         
        c_skip = (alpha_t * alpha_s + sigma_t * sigma_s) / (alpha_t**2 + sigma_t**2)

        c_out = - (alpha_s * sigma_t - alpha_t * sigma_s) * (alpha_t**2 + sigma_t**2).rsqrt() * self.sigma_data
        
        return c_skip * yt + c_out * y
    
    def euler_fm_sample_function(self, yt, y, t, s ):
        assert self.noise_schedule == 'fm'  

        
        return  yt - (t - s) * self.sigma_data *  y 
          
    def nt_to_t(self, nt):
        dtype = nt.dtype
        nt = nt.to(torch.float64)
        if self.noise_schedule == "vp_cosine":
            t = torch.arctan(nt) / (torch.pi * 0.5) 
 
        elif self.noise_schedule == "fm":
            t = nt / (1 + nt)
            
        t = torch.nan_to_num(t, nan=1)

        t = t.to(dtype)
            

        if (
            self.noise_schedule.startswith("vp")
            and self.noise_schedule == "fm"
            and t.max() > 1
        ):
            raise ValueError(f"t out of range: {t.min().item()}, {t.max().item()}")
        return t

    def get_init_noise(self, shape, device):
        
        noise = torch.randn(shape, device=device) * self.sigma_data
        return noise

    def forward_model(
        self,
        model,
        x,
        t,
        s,
        class_labels=None, 
        force_fp32=False,
        **model_kwargs,
    ):
 
              
        
        alpha_t, sigma_t = self.get_alpha_sigma(t)
    
        c_in = (alpha_t ** 2 + sigma_t**2 ).rsqrt() / self.sigma_data  
        if self.temb_type == 'identity': 

            c_noise_t = t  * self.time_scale
            c_noise_s = s  * self.time_scale
            
        elif self.temb_type == 'stride':

            c_noise_t = t * self.time_scale
            c_noise_s = (t - s) * self.time_scale
            
        with torch.amp.autocast('cuda', enabled=self.use_mixed_precision   and not force_fp32, dtype= self.mixed_precision ):
            F_x = model( 
                (c_in * x) ,
                c_noise_t.flatten() ,
                c_noise_s.flatten() ,
                class_labels=class_labels, 
                **model_kwargs,
            )   
        return F_x

    
    def forward(
        self,
        x,
        t,
        s=None, 
        class_labels=None, 
        force_fp32=False, 
        **model_kwargs,
    ):
        dtype = t.dtype  
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        ) 
            
        F_x = self.forward_model(
            self.model,
            x.to(torch.float32),
            t.to(torch.float32).reshape(-1, 1, 1, 1),
            s.to(torch.float32).reshape(-1, 1, 1, 1) if s is not None else None,
            class_labels, 
            force_fp32,
            **model_kwargs,
        ) 
        F_x = F_x.to(dtype) 
         
        if self.f_type == "identity":
            F_x  =  self.ddim(x, F_x , t, s)  
        elif self.f_type == "simple_edm": 
            F_x = self.simple_edm_sample_function(x, F_x , t, s)   
        elif self.f_type == "euler_fm": 
            F_x = self.euler_fm_sample_function(x, F_x, t, s)  
        else:
            raise NotImplementedError
 
        return F_x
 
    def cfg_forward(
        self,
        x,
        t,
        s=None, 
        class_labels=None,
        force_fp32=False,
        cfg_scale=None, 
        **model_kwargs,
    ):
        dtype = t.dtype   
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        ) 
        if cfg_scale is not None: 

            x_cfg = torch.cat([x, x], dim=0) 
            class_labels = torch.cat([torch.zeros_like(class_labels), class_labels], dim=0)
        else:
            x_cfg = x 
        F_x = self.forward_model(
            self.model,
            x_cfg.to(torch.float32),
            t.to(torch.float32).reshape(-1, 1, 1, 1) ,
            s.to(torch.float32).reshape(-1, 1, 1, 1)  if s is not None else None,
            class_labels=class_labels,
            force_fp32=force_fp32,
            **model_kwargs,
        ) 
        F_x = F_x.to(dtype) 
        
        if cfg_scale is not None: 
            uncond_F = F_x[:len(x) ]
            cond_F = F_x[len(x) :] 
            
            F_x = uncond_F + cfg_scale * (cond_F - uncond_F) 
         
        if self.f_type == "identity":
            F_x =  self.ddim(x, F_x, t, s)  
        elif self.f_type == "simple_edm": 
            F_x  = self.simple_edm_sample_function(x, F_x , t, s)   
        elif self.f_type == "euler_fm": 
            F_x = self.euler_fm_sample_function(x, F_x , t, s)  
        else:
            raise NotImplementedError

        return F_x



    ######################################################### for training


    def get_logsnr_prime(self, t):
        if self.noise_schedule == "vp_cosine":
            return (
                -1
                * torch.pi
                / (torch.sin(t * torch.pi * 0.5) * torch.cos(t * torch.pi * 0.5))
            )
 
        elif self.noise_schedule == "fm":
            return -2 * (1 / (1 - t) / t)

            
    def get_kernel_weight(
        self, t,  s, a=1,   b=0, 
    ): 
             
        logsnr_t = self.get_logsnr(t)

        alpha_t, sigma_t = self.get_alpha_sigma(t) 
        alpha_s, sigma_s = self.get_alpha_sigma(s) 
         

        if self.f_type == 'identity':
            w =   (sigma_t / (alpha_s * sigma_t - sigma_s * alpha_t )).abs()
        elif self.f_type == 'simple_edm':
            w =  (alpha_t**2 + sigma_t**2).sqrt() / (alpha_s * sigma_t - alpha_t * sigma_s).abs() / self.sigma_data
        elif self.f_type == 'euler_fm':
            w = 1 / (t - s).abs() / self.sigma_data
        else:
            raise NotImplementedError 


        neg_dlogsnr_dt = - self.get_logsnr_prime(t)
                
        wout =  alpha_t ** a / (alpha_t**2 + sigma_t**2)  * 0.5 * neg_dlogsnr_dt * (b - logsnr_t).sigmoid()
         
        return   w, wout

    def sample_eta_t(
        self,
        batch_size,
        device,
        log_low=None,
        log_high=None,
        low=None,
        high=None,
        sample_mode="lognormal",
        P_mean=-1.1,
        P_std=2.0,
        **kwargs,
    ):
        if sample_mode == "lognormal":

            log_low = log_low if log_low is not None else -float("inf")
            log_low = torch.as_tensor(log_low, device=device, dtype=torch.float64)
            if log_low.ndim == 0:
                log_low = log_low.unsqueeze(0).expand(batch_size).reshape(-1, 1, 1, 1)

            log_high = log_high if log_high is not None else float("inf")
            log_high = torch.as_tensor(log_high, device=device, dtype=torch.float64)
            if log_high.ndim == 0:
                log_high = log_high.unsqueeze(0).expand(batch_size).reshape(-1, 1, 1, 1)
            dist = torch.distributions.Normal(
                loc=torch.full_like(log_high, P_mean),
                scale=torch.full_like(log_high, P_std),
            )

            cdf = torch.rand([batch_size, 1, 1, 1], device=device) * (
                dist.cdf(log_high) - dist.cdf(log_low)
            ) + dist.cdf(log_low)

            log_nt = dist.icdf(cdf)
            nt = log_nt.exp()

        elif sample_mode == "uniform":
            if high is None:
                high = self.nt_high
            else:
                high = torch.as_tensor(high, device=device, dtype=torch.float64)
                if high.ndim == 0:
                    high = high.unsqueeze(0).expand(batch_size).reshape(-1, 1, 1, 1)

            if low is None:
                low = self.nt_low
            else:
                low = torch.as_tensor(low, device=device, dtype=torch.float64)
                if low.ndim == 0:
                    low = low.unsqueeze(0).expand(batch_size).reshape(-1, 1, 1, 1)

            high_t = self.nt_to_t(high)
            low_t = self.nt_to_t(low)

            t = (
                torch.rand([batch_size, 1, 1, 1], device=device, dtype=torch.float64) * (high_t - low_t)
                + low_t
            )

            log_nt = self.get_log_nt(t)
            nt = log_nt.exp()
        else:
            raise ValueError(f"Unknown sample_t_mode: {sample_mode}")

        return nt, log_nt
