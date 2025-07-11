 
import torch 
from torch_utils import persistence 

from einops import rearrange, einsum

import numpy as np 
 
   
    
     
@persistence.persistent_class
class IMMLoss:

    def __init__(
        self,  
        sigma=1, 
        sample_t_mode="lognormal",
        P_mean=-1.1,
        P_std=2.0, 
        matrix_size=4, 
        sample_repeat=1,
        label_dropout=0.1, 
        k=12,
        a=2,
        b=4, 
        min_tr_gap=None,  
        **kwargs,
    ):  
        super().__init__()

        self.P_mean = P_mean
        self.P_std = P_std
         
        self.min_tr_gap = min_tr_gap
          
 
        self.sigma = sigma
 
 
        self.sample_t_mode = sample_t_mode 
 


        self.matrix_size = matrix_size
         
        
        self.sample_repeat = sample_repeat
         
        
        self.label_dropout = label_dropout
         
         
        ####
        self.a = a
        self.b = b

        self.k = k
         
            
    def nt_to_nr(self, nt, net):

        u = (net.module.nt_high - net.module.nt_low) * (1/ 2) ** self.k
         
        nr = (nt -  u ).clamp(min=net.module.nt_low, max=net.module.nt_high) 
         
        return nr
     
    
    def kernel_fn(self, x, y,   flatten_dim,   w ):
        
        loss = (
                torch.clamp_min(
                    ((x - y) ** 2).flatten(flatten_dim).sum(-1)  , 1e-8
                )
            ).sqrt()   / (np.prod(y.shape[flatten_dim:])) / self.sigma
            
            
        ret = torch.exp(  -loss * w ) 
        return ret 
     
    def kernel(
        self,
        x,
        y,   
        w=None, 
    ):

        # x: (t, b, ...)
        # y: (t, b, ...)

        x = x.unsqueeze(2)  # (t, b, 1, ...)
        y = y.unsqueeze(1)  # (t, 1, b, ...)
         
        if w is None:
            w = 1
        else:
            w = w[:, None, None]
  
            
        ret = self.kernel_fn(x, y,  flatten_dim=3,  w=w )

        return ret

    def sample_trs(self, t_bs, net, device ):

        high = net.module.nt_high 
        low = net.module.nt_low 
        nt, log_nt = net.module.sample_eta_t(
            t_bs,
            device,
            log_low=np.log(low),
            log_high=np.log(high),
            low=low,
            high=high,
            sample_mode=self.sample_t_mode,
            P_mean=self.P_mean,
            P_std=self.P_std,
        ) 
 
        ns_upper = nt
        logns_upper = log_nt
          
        ns, log_ns = net.module.sample_eta_t(
            t_bs,
            device,
            log_low=np.log(net.module.nt_low),
            log_high=logns_upper,
            low=net.module.nt_low,
            high=ns_upper,
            sample_mode=self.sample_t_mode,
            P_mean=self.P_mean,
            P_std=self.P_std,
        ) 
        ns = torch.minimum(ns, nt).clamp(min=net.module.nt_low)
                
        
        nr = self.nt_to_nr(nt, net ) 
                   
            
        t = net.module.nt_to_t(nt) 
        r = net.module.nt_to_t(nr)
        s = net.module.nt_to_t(ns)
            
        assert torch.allclose(net.module.get_log_nt(t).exp(), nt)
        assert torch.allclose(net.module.get_log_nt(r).exp(), nr)
        assert torch.allclose(net.module.get_log_nt(s).exp(), ns)
 
        if self.min_tr_gap is not None: 
            max_r = torch.clamp(t - self.min_tr_gap, min=net.module.nt_low,  )
         
            r  = torch.minimum(r , max_r ) 
        # makes sure s<r<t  
        r  = torch.maximum(r , s ).clamp(min=net.module.eps)  
            
         
             
        
        return t, r, s

    def process_labels(self, labels):
        if labels is not None:
            labels_clone = labels.clone()
            mask = torch.rand(labels.shape[0]) < self.label_dropout
            labels_clone[mask] = 0
            return labels_clone, mask
        else:
            return labels, None
      
     
    def get_loss(self,  f_st, f_sr,   w, wout,    ):
          
        # MMD 
        inter_sample_sim = self.kernel(
            f_st ,
            f_st , 
            w=w,
        ) 
            

        inter_gt_sim = self.kernel(
            f_sr ,
            f_sr , 
            w=w,
        )  
        
            
        cross_sim = self.kernel(
            f_st ,
            f_sr , 
            w=w,
        ) 
             
        inter_sample_sim =inter_sample_sim.mean((1, 2)) 
        cross_sim = cross_sim.mean((1, 2))
        inter_gt_sim = inter_gt_sim.mean((1, 2))
            
        loss = inter_sample_sim + inter_gt_sim - 2 * cross_sim 
            
        if wout is not None:
            loss = wout * loss
               
        logs = {
            "inter_sample_sim": inter_sample_sim.detach(),
            "inter_gt_sim": inter_gt_sim.detach(),
            "cross_sim": cross_sim.detach(),
        }
        return loss, logs
     
    
    def __call__(
        self,
        net,
        images,
        labels=None,
        augment_pipe=None, 
        device=torch.device("cuda"), 
        **kwargs
    ): 
        
        images = images.to(device)
        labels = labels.to(device) if labels is not None else None
         
        
        current_matrix_size = self.matrix_size 
        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)

        t, r, s = self.sample_trs(images.shape[0]  // current_matrix_size, net, device=images.device, )
 
        t = t.repeat_interleave(current_matrix_size, dim=0)
        s = s.repeat_interleave(current_matrix_size, dim=0)
        r = r.repeat_interleave(current_matrix_size, dim=0)   

        # Augmentation if needed
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        y = y.repeat_interleave(self.sample_repeat, dim=0)
        augment_labels = augment_labels.repeat_interleave(self.sample_repeat, dim=0) if augment_labels is not None else None
        labels_drop, drop_mask = self.process_labels(labels) 
        labels = labels.repeat_interleave(self.sample_repeat, dim=0) if labels is not None else None
        labels_drop = labels_drop.repeat_interleave(self.sample_repeat, dim=0) if labels_drop is not None else None
         
 
        yt, noiset = net.module.add_noise(y, t)
          
 
        yr = net.module.ddim(yt, y , t, r, noise= noiset)
                 

        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
         

        f_st = net(
            yt,
            t,
            s, 
            labels_drop, 
            augment_labels=augment_labels,  
        ) 
         
        torch.cuda.set_rng_state(rng_state) 
        with torch.no_grad() :   
            f_sr = net(
                yr,
                r,
                s, 
                labels,  
                augment_labels=augment_labels, 
                force_fp32=True,   
            ) 
               
             
        f_st = rearrange(f_st, "(b m) ... -> b m ...", m=current_matrix_size)
        f_sr = rearrange(f_sr, "(b m) ... -> b m ...", m=current_matrix_size) 
        yt = rearrange(yt, "(b m) ... -> b m ...", m=current_matrix_size)
        yr = rearrange(yr, "(b m) ... -> b m ...", m=current_matrix_size)
        t = rearrange(t, "(b m) ... -> b m ...", m=current_matrix_size)
        r = rearrange(r, "(b m) ... -> b m ...", m=current_matrix_size) 
        s = rearrange(s, "(b m) ... -> b m ...", m=current_matrix_size)
          
             
         
        wt, wtout = net.module.get_kernel_weight(
            t[:, 0].flatten(), 
            s[:, 0].flatten(),   
            a=self.a, 
            b=self.b, 
        )    
                 
        loss, loss_logs = self.get_loss( f_st, f_sr ,  wt, wtout  )
              
        logs = {
            "r_t_ratio": r[:, 0] / t[:, 0], 
            "s_t_ratio": s[:, 0] / t[:, 0],
            "t_r_diff": t[:, 0] - r[:, 0],
            'loss': loss,
            **loss_logs
        } 

        if torch.isnan(loss).any():
            print("Nan in loss")
            loss = torch.nan_to_num(loss) 
 
        logs["ts"] = t[:, 0].flatten()  
                   
        return loss, logs



 












 
  




 