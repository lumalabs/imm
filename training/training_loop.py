import os
import time
import copy
import json
import pickle
import psutil
import functools
import PIL.Image
import numpy as np
import torch
import dnnlib  
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc  

from metrics import metric_main

import wandb
from einops import rearrange

from omegaconf import DictConfig, OmegaConf, ListConfig

# ----------------------------------------------------------------------------


def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 16)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 16)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [
                indices[(i + gw) % len(indices)] for i in range(len(indices))
            ]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


# ----------------------------------------------------------------------------


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], "L").save(fname)
    if C == 3:
        PIL.Image.fromarray(img, "RGB").save(fname)


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
 
  
 

def training_loop(
    config: DictConfig,
    resume_pkl=None,
    resume_tick=None,
    resume_state_dump=None,
    device=torch.device("cuda"),
    run_dir=".",  # Output directory.
):
    seed = config.training.seed
    cudnn_benchmark = config.training.cudnn_benchmark
    enable_tf32 = config.training.enable_tf32 
    total_ticks = config.training.total_ticks
    kimg_per_tick = config.training.kimg_per_tick 
    ema_halflife_kimg = config.training.ema_halflife_kimg
    ema_beta = config.training.ema_beta 
    ema_rampup_ratio = config.training.get("ema_rampup_ratio", None)
 
    ckpt_ticks = config.training.ckpt_ticks
    snapshot_ticks = config.training.snapshot_ticks
    state_dump_ticks = config.training.state_dump_ticks
    sample_ticks = config.training.sample_ticks
    eval_ticks = config.training.eval_ticks

    batch_size = config.training.batch_size
    batch_gpu = config.training.batch_gpu

    metrics = config.training.metrics

    dataset_kwargs = config.dataset
    data_loader_kwargs = config.dataloader
    network_kwargs = config.network
    encoder_kwargs = config.encoder 
    loss_kwargs = config.loss
    optimizer_kwargs = config.optimizer
    augment_kwargs = config.augment 
    
    sample_kwargs_dict = config.get('sampling', {}) 
    mid_nt = sample_kwargs_dict.get('few_step', {'mid_nt': [0.821]})['mid_nt']

    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31)) 
    torch.backends.cudnn.benchmark = cudnn_benchmark 

    # Enable these to speed up on A100 GPUs
    print("enable_tf32", enable_tf32)
    torch.backends.cudnn.allow_tf32 = enable_tf32
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32 
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0("Loading dataset...")
    dataset_obj = dnnlib.util.construct_class_by_name(
        **dataset_kwargs
    )  # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed,
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu  ,
            **data_loader_kwargs,
        )
    )
    grid_size, images, labels = setup_snapshot_image_grid(training_set=dataset_obj)
    
    # Setup encoder
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs) 

    ref_image = encoder.encode(torch.as_tensor(images[:1]).to(device))
    augment_pipe = (
        dnnlib.util.construct_class_by_name(**augment_kwargs)
        if augment_kwargs is not None
        else None
    )  # training.augment.AugmentPipe
    # Construct network.
    dist.print0("Constructing network...")
    interface_kwargs = dict(
        img_resolution=dataset_obj.resolution,
        img_channels=ref_image.shape[1],
        label_dim=dataset_obj.label_dim,
        num_classes=dataset_obj.label_dim,
    ) 
    net = dnnlib.util.construct_class_by_name(
        **network_kwargs, **interface_kwargs
    )  # subclass of torch.nn.Module
     
    net.train().requires_grad_(True).to(device)  
     
    # Setup optimizer.
    dist.print0("Setting up optimizer...")
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs, device=device, vae=encoder) 
    

    dist.print0("Setting up DDP...")
    ddp = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[device], 
    )
    ema = copy.deepcopy(net).eval().requires_grad_(False)
       
    optimizer = dnnlib.util.construct_class_by_name(
        params=ddp.parameters(), **optimizer_kwargs
    )  # subclass of torch.optim.Optimizer

     
        
    total_ksteps = total_ticks*kimg_per_tick//batch_size 
    
    scaler = torch.amp.GradScaler('cuda', enabled=config.network.get('mixed_precision', None) == 'fp16')
    # Resume training from previous snapshot. 
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first 
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f) 
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow  
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=net, require_all=False
        )
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=ema, require_all=False
        )
        del data  # conserve memory
        
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device("cpu"))
        misc.copy_params_and_buffers(
            src_module=data["net"], dst_module=net, require_all=True
        ) 
        optimizer.load_state_dict(data["optimizer_state"]) 
        if 'scaler_state' in data:
            scaler.load_state_dict(data['scaler_state'])
        del data  # conserve memory
         


    if dist.get_rank() == 0:
        if config.logger.get("api_key", None):
            wandb.login(key=config.logger.pop("api_key"))
        wandb.init(
            **config.logger,
            name=config.name,
            config=OmegaConf.to_container(config),
        )
    # Export sample images.
    grid_z = None
    grid_c = None

    mid_t = [net.nt_to_t(torch.as_tensor(nt)).item() for nt in mid_nt] 
 
    if config.training.get("debug", False):
        images, labels = next(dataset_iterator)
        images = encoder.encode(images.to( device) ) 
        labels = labels.to(device)
        
    if dist.get_rank() == 0:

        if config.training.get("debug", False):
 
            if dist.get_rank() == 0:
                dist.print0("Exporting sample images...")

                num_samples = config.training.get("num_samples", min(images.shape[0], 32))

                grid_z = net.get_init_noise(
                    [
                        num_samples,
                        ema.img_channels,
                        ema.img_resolution,
                        ema.img_resolution,
                    ],
                    device,
                )

                mid_grid_z = net.add_noise(
                    images[:num_samples],
                    torch.tensor(mid_t, device=device),
                )[0]

                wandb.log(
                    {
                        "mid_init_image": [
                            wandb.Image(
                                rearrange(
                                   dec_mid_grid_z,
                                    "c h w -> h w c",
                                ),
                            )
                            for dec_mid_grid_z in encoder.decode(mid_grid_z[:num_samples] ).detach().cpu().numpy()
                        ],
                        "mid_groundtruth": [
                            wandb.Image(
                                rearrange(
                                    dec_image,
                                    "c h w -> h w c",
                                ),
                            )
                            for dec_image in  encoder.decode(images[:num_samples] ).detach().cpu().numpy()
                        ],
                    },
                    step=0,
                )

                grid_z = grid_z.split(batch_gpu)
                mid_grid_z = mid_grid_z.split(batch_gpu)

                mid_grid_c = torch.as_tensor(labels, device=device)[:num_samples]
                mid_grid_c = mid_grid_c.split(batch_gpu) 

                grid_c = torch.nn.functional.one_hot(torch.randint(dataset_obj.label_dim, (labels.shape[0],), device=device)[:num_samples], num_classes=dataset_obj.label_dim) if dataset_obj.has_labels else torch.as_tensor(labels, device=device)[:num_samples]
                grid_c = grid_c.split(batch_gpu) 
 
        else:
 
            dist.print0("Exporting sample images...")

            num_samples = config.training.get("num_samples", min(images.shape[0], 32))

            grid_z = net.get_init_noise(
                [num_samples, ema.img_channels, ema.img_resolution, ema.img_resolution],
                device,
            ) 

            mid_grid_z = net.add_noise(
                encoder.encode(torch.as_tensor(images[:num_samples], device=device) ),
                torch.tensor(mid_t, device=device),
            )[0]
            if dist.get_rank() == 0:
                wandb.log(
                    {
                        "mid_init_image": [
                            wandb.Image(
                                rearrange(
                                    dec_mid_grid_z,
                                    "c h w -> h w c",
                                ),
                            )
                            for dec_mid_grid_z in encoder.decode(mid_grid_z[:num_samples] ).detach().cpu().numpy()
                        ],
                        "mid_groundtruth": [
                            wandb.Image(
                                rearrange(
                                    dec_image,
                                    "c h w -> h w c",
                                ),
                            )
                            for dec_image in encoder.decode(encoder.encode(torch.as_tensor(images[:num_samples], device=device) ) ).detach().cpu().numpy()
                        ],
                    },
                    step=0,
                )

            grid_z = grid_z.split(batch_gpu)
            mid_grid_z = mid_grid_z.split(batch_gpu)
            
            mid_grid_c = torch.as_tensor(labels, device=device)[:num_samples]
            mid_grid_c = mid_grid_c.split(batch_gpu) 

            grid_c = torch.nn.functional.one_hot(torch.randint(dataset_obj.label_dim, (labels.shape[0],), device=device)[:num_samples], num_classes=dataset_obj.label_dim) if dataset_obj.has_labels else torch.as_tensor(labels, device=device)[:num_samples]
            grid_c = grid_c.split(batch_gpu) 
    
    # Train.
    dist.print0(f"Training for {total_ksteps}k iter...")
    dist.print0()
    cur_nimg = resume_tick * kimg_per_tick * 1000
    cur_tick = resume_tick
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg / 1000, total_ticks*kimg_per_tick)
 
 

    cur_tick += 1 
    generator_fn_dict = {k: functools.partial(generator_fn, **sample_kwargs) for k, sample_kwargs in sample_kwargs_dict.items()}
       
    global_step = 0
     
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True) 
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                if not config.training.get("debug", False):
                    images, labels = next(dataset_iterator)
                    images = encoder.encode(images.to(device) )
                    labels = labels.to(device) 
                     
                loss, logs = loss_fn(
                    net=ddp,
                    images=images,
                    labels=labels,
                    augment_pipe=augment_pipe,   
                    device=device,
                )      
                
                ts = logs.pop("ts") 
                for k, v in logs.items():
                    training_stats.report(
                        f"Loss/{k}",
                        torch.nan_to_num(v, nan=0, posinf=1e5, neginf=-1e5),
                        ts=ts,
                        max_t=net.T,
                        num_bins=4,
                    )  
                scaler.scale(loss.mean()).backward()  
                
        scaler.step(optimizer) 
        scaler.update() 
            
            
        # Update EMA.
        if ema_halflife_kimg is not None:
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
              
        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = cur_tick >= total_ticks

        global_step += 1
        if (
            (not done)
            and (cur_tick != 0)
            and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000)
        ):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"
        ] 
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(" ".join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0("Aborting...")

        # Save network snapshot.
        if (
            (snapshot_ticks is not None)
            and (done or (isinstance(snapshot_ticks, ListConfig) and cur_tick in snapshot_ticks) or (isinstance(snapshot_ticks, int)   and cur_tick % snapshot_ticks == 0))
            and cur_tick != 0
        ): 
            data = dict(
                ema=ema,  
                augment_pipe=augment_pipe,
                dataset_kwargs=dict(dataset_kwargs),
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.get_rank() == 0:
                try:
                    with open(
                        os.path.join(run_dir, f"network-snapshot-{cur_tick:06d}.pkl"), "wb"
                    ) as f:
                        pickle.dump(data, f)
                except Exception as e:
                    dist.print0(f"Failed to save the snapshot: {e}")
            del data  # conserve memory 

        # Save full dump of the training state.
        if (
            (state_dump_ticks is not None)
            and (done or cur_tick % state_dump_ticks == 0)
            and cur_tick != 0
            and dist.get_rank() == 0
        ):  
            try: 
                torch.save(
                    dict(net=net,  
                        optimizer_state=optimizer.state_dict(),  
                         scaler_state=scaler.state_dict()
                        ),
                    os.path.join(run_dir, f"training-state-{cur_tick:06d}.pt"),
                )
            except Exception as e:
                dist.print0(f"Failed to save the training state: {e}")

        # Save latest checkpoints
        if (
            (ckpt_ticks is not None)
            and (done or cur_tick % ckpt_ticks == 0)
            and cur_tick != 0
        ): 
            dist.print0(f"Save the latest checkpoint at {cur_tick:06d} img...")
             
            data = dict(
                ema=ema, 
                augment_pipe=augment_pipe,
                dataset_kwargs=dict(dataset_kwargs),
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.get_rank() == 0:
                try:
                    with open(
                        os.path.join(run_dir, f"network-snapshot-latest.pkl"), "wb"
                    ) as f:
                        pickle.dump(data, f)
                except Exception as e:
                    dist.print0(f"Failed to save the latest snapshot: {e}")
            del data  # conserve memory 

            if dist.get_rank() == 0:
                try:
                    if config.training.use_zero:
                        optimizer.consolidate_state_dict()
                    torch.save(
                        dict(net=net,   optimizer_state=optimizer.state_dict(), 
                             scaler_state=scaler.state_dict()
                            ),
                        os.path.join(run_dir, f"training-state-latest.pt"),
                    )
                except Exception as e:
                    dist.print0(f"Failed to save the latest checkpoint: {e}")
                     
        # Evaluation
        if (eval_ticks is not None) and (done or (cur_tick % eval_ticks == 0 and cur_tick != 0)):
            dist.print0("Evaluating models...")
            for metric in metrics:
                
                for key, gen_fn in generator_fn_dict.items():
                    result_dict = metric_main.calc_metric(
                        metric=metric,
                        generator_fn=gen_fn,
                        decode_fn=encoder.decode,
                        G=ema,
                        G_kwargs={},
                        dataset_kwargs=dataset_kwargs,
                        detector_kwargs=config.eval.get('detector_kwargs', {}),
                        detector_url=config.eval.detector_url,
                        num_gpus=dist.get_world_size(),
                        rank=dist.get_rank(),
                        device=device,
                        ref_stats=config.eval.get('ref_stats', None),
                    )
                    if dist.get_rank() == 0: 
                        wandb.log( {f"{key}_{k}": v for k, v in result_dict['results'].items()},
                                step=cur_tick,)
                        os.makedirs(os.path.join(run_dir, key), exist_ok=True)
                        print('FID:', key)
                        metric_main.report_metric(
                            result_dict,
                            run_dir=os.path.join(run_dir, key),
                            snapshot_pkl="network-snapshot-latest.pkl",
                        )
  
        # Sample Img
        if (
            (sample_ticks is not None)
            and (done or cur_tick % sample_ticks == 0)
            and dist.get_rank() == 0
        ):
            dist.print0("Exporting sample images...")
            for grid_z_, grid_c_,   name in zip(
                [grid_z, mid_grid_z], [grid_c, mid_grid_c],  ["uncond", "mid"]
            ):
                res = {}
                
                for key, gen_fn in generator_fn_dict.items():
                    if name == 'mid' and key != '1_step':
                        continue   
                    samples = [
                        gen_fn(
                            ema if not config.training.get("debug", False) else net,
                            z,
                            c, 
                        ) 
                        .reshape(*z.shape)
                        for z, c in zip(grid_z_, grid_c_)
                    ]
                    samples = torch.cat(samples)
                    res[name + '_' + key] = samples
                 
                if dataset_obj.has_labels:
                    labels_idx = torch.cat(grid_c_).argmax(dim=1)
                    labels_idx[torch.cat(grid_c_).sum(dim=1) == 0] = dataset_obj.label_dim
                else:
                    labels_idx = None
                wandb.log(
                    {
                        f"{key}_sample": [
                            wandb.Image(
                                rearrange(
                                    dec_sample,
                                    "c h w -> h w c",
                                ),
                                caption=f"label: {labels_idx[ii] if labels_idx is not None else 0},  max: {dec_sample.max():.2f}, min: {dec_sample.min():.2f}, std: {dec_sample.std():.2f}, mean: {dec_sample.mean():.2f}",
                            )
                            for ii, dec_sample in enumerate( encoder.decode(samples[:32] ).detach().cpu().numpy())
                        ] for key, samples in res.items()
                    },
                    step=cur_tick,
                )

            del res
          
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            
            logs = {
                    k: v["mean"]
                    for k, v in training_stats.default_collector.as_dict().items()
                } 

            wandb.log(
                logs,
                step=cur_tick,
            ) 
        dist.update_progress(cur_nimg / 1000, total_ticks*kimg_per_tick)
        # Update state.
        cur_tick += 1

        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

 


    # Few-step Evaluation. 

    if dist.get_rank() == 0:
        dist.print0("Exporting final sample images...")
        res = {}
        for key, gen_fn in generator_fn_dict.items():
        
            images = [gen_fn(ema, z, c).reshape(*z.shape) for z, c in zip(grid_z, grid_c)]
            images = torch.cat(images) 
            res[key] = images
        wandb.log(
            {
                "final_" + key: [
                    wandb.Image(
                        rearrange(
                            dec_sample,
                            "c h w -> h w c",
                        )
                    )
                    for dec_sample in  encoder.decode(images[:32] ).detach().cpu().numpy()
                ] for key, images in res.items()
            }
        )
        del images

    dist.print0("Evaluating few-step generation...")
    for _ in range(3):
        for metric in metrics:
            for key, gen_fn in generator_fn_dict.items():
                result_dict = metric_main.calc_metric(
                    metric=metric,
                    generator_fn=gen_fn,
                    decode_fn=encoder.decode,
                    G=ema,
                    G_kwargs={},
                    dataset_kwargs=dataset_kwargs,
                    detector_kwargs=config.eval.get('detector_kwargs', {}),
                    detector_url=config.eval.detector_url,
                    num_gpus=dist.get_world_size(),
                    rank=dist.get_rank(),
                    device=device,
                    ref_stats=config.training.get('ref_stats', None),
                )
                if dist.get_rank() == 0: 
                    wandb.log( {f"{key}_{k}": v for k, v in result_dict['results'].items()},
                            step=cur_tick,)
                    os.makedirs(os.path.join(run_dir, key), exist_ok=True)
                    print('FID:', key)
                    metric_main.report_metric(
                        result_dict,
                        run_dir=os.path.join(run_dir, key),
                        snapshot_pkl="network-snapshot-latest.pkl",
                    )

    # Done.
    dist.print0()
    dist.print0("Exiting...")


# ----------------------------------------------------------------------------
