# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import GPUtil
import torch
import numpy as np
from diffu_models.precond import EDMPrecond, VEPrecond, iDDPMPrecond
from diffu_models.dit_models import DiT_XL_8,  DiT_XL_4, DiT_XL_2, DiT_L_8, DiT_L_4, DiT_L_2, DiT_B_8, DiT_B_4, DiT_B_2, DiT_S_8, DiT_S_4, DiT_S_2, DiT_L_5, DiT_B_5
from diffu_models.losses import EDMLoss, VELoss, VPLoss
import argparse
import yaml
import edm.dnnlib as dnnlib
from edm.torch_utils import distributed as dist
from edm.torch_utils import training_stats
from edm.torch_utils import misc
import re
from utils.utils_func import desc_from_config
from fmap_data.fmap_dataset import FmapLiveTemplateDataset
#----------------------------------------------------------------------------

torch.backends.cudnn.enabled = False
DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8, 'DiT-L/5': DiT_L_5,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8, 'DiT-B/5': DiT_B_5,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


def training_loop(cfg):
    run_dir             = cfg['data']['run']      # Output directory.
    #network      = None,       # Options for model and preconditioning.
    #seed                = 0,        # Global random seed.
    batch_size          = cfg['hyper_params']['batch_size']    # Total batch size for one training iteration.
    state_dump_ticks    = 500     # How often to dump training state, None = disable.
    resume_pkl          = None    # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None    # Start from the given training state, None = reset training state.
    resume_kimg         = cfg['resume']['resume_kimg']        # Start from the given training progress.
    cudnn_benchmark = True  # Enable torch.backends.cudnn.benchmark?
    device = cfg['device']
    gpus = GPUtil.getGPUs()
    gpu = gpus[int(device[-1])]
    # Initialize.
    start_time = time.time()
    #np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))iDDPMPrecond
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    # if batch_gpu is None or batch_gpu > batch_gpu_total:
    batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    print(f"Will accumulate for {num_accumulation_rounds} rounds")
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')

    dataset_obj = FmapLiveTemplateDataset(cfg["data"]["root_dir"],  absolute=cfg["data"]["abs"])
    collate_fn = None
    loader = torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=batch_gpu, collate_fn=collate_fn, num_workers=0, shuffle=True)

    # Construct network.
    dist.print0('Constructing network...')
    

    num_coeffs = cfg["data"]["n_fmap"]
    raw_net = DiT_models[cfg["architecture"]["name_arch"]](conditioning=cfg["architecture"]["cond"], input_size=num_coeffs, in_channels=1, learn_sigma=False)
    # if conditioning:
    #     raw_net = ScoreModelFCCond(cfg, n_coeffs=num_coeffs, hidden_dim=cfg["architecture"]["hidden_dim"],
    #                            embed_dim=cfg["architecture"]["embed_dim"], c_dim=cond_size)
    # else:
    
    raw_net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, 1, num_coeffs, num_coeffs], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            input_batch = [images, sigma]
            # if conditioning:
            #     latents = torch.zeros([batch_gpu, cond_size], device=device)
            #     misc.print_module_summary(raw_net, [images, sigma, latents], max_nesting=2)
            # else:
            misc.print_module_summary(raw_net, input_batch, max_nesting=2)
    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_name = cfg['hyper_params']['loss_name']
    loss_fn = None
    if loss_name == "EDMLoss":
        loss_fn = EDMLoss()
        net = EDMPrecond(raw_net)#, label_dim=cond_size)
    elif loss_name == "VPLoss":
        loss_fn = VPLoss()
        net = iDDPMPrecond(raw_net).to(device)#, label_dim=cond_size)
    elif loss_name == "VELoss":
        loss_fn = VELoss()
        net = VEPrecond(raw_net)#, label_dim=cond_size)

    lr = float(cfg['hyper_params']['lr'])
    print("PARAM")
    counting = 0
    for param in list(net.parameters()):
        print(counting, param.shape)
        counting +=1
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, betas=[0.9, 0.99], eps=1e-8) # subclass of torch.optim.Optimizer
    #augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe

    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if cfg["resume"]["pkl"] is not None:
        resume_pkl = os.path.join(cfg["data"]["run"], cfg["resume"]["pkl"])
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
        resume_state_dump = os.path.join(cfg["data"]["run"], f'training-state-{cfg["resume"]["pkl"].split(".")[0].split("-")[-1]}.pt')
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    total_shapes = len(dataset_obj)
    dist.print0(f'Training for {total_shapes} shapes...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_shapes)
    stats_jsonl = None
    count_accumulation = 0
    for i in range(1000):
        for data in loader:
            #for shapes in dataloader:
            # Accumulate gradients.
            if count_accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
            with misc.ddp_sync(ddp, (count_accumulation == num_accumulation_rounds - 1)):
                # if conditioning:
                #     shapes, latent = data
                #     #shapes = normalize(shapes, mean, std)[:, -num_coeffs:]
                #     shapes = shapes.to(device).to(torch.float32)
                #     latent = latent.to(device).to(torch.float32)
                #     latent = latent * (
                #                 torch.rand([latent.shape[0], 1], device=latent.device) >= cfg["hyper_params"][
                #             "cond_dropout"]).to(latent.dtype)
                #     #pdb.set_trace()
                #     loss = loss_fn(net=ddp, x=shapes, latents=latent)
                # else:
                shapes = data
                shapes = shapes.to(device).to(torch.float32)
                loss = loss_fn(net=ddp, x=shapes, latents=None)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(cfg["hyper_params"]["ls"] / batch_gpu_total).backward()


            if count_accumulation == num_accumulation_rounds - 1:
                # Update weights.
                for g in optimizer.param_groups:
                    g['lr'] = cfg["hyper_params"]["lr"] * min(cur_nimg / max(cfg["hyper_params"]["lr_rampup_kimg"] * 1000, 1e-8), 1)
                for param in net.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                optimizer.step()

                # Update EMA.
                ema_halflife_nshape = cfg["hyper_params"]["ema_halflife_nshape"] * 1000
                if cfg["hyper_params"]["ema_rampup_ratio"] is not None:
                    ema_halflife_nimg = min(ema_halflife_nshape, cur_nimg * cfg["hyper_params"]["ema_rampup_ratio"])
                ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
                for p_ema, p_net in zip(ema.parameters(), net.parameters()):
                    p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

            # Perform maintenance tasks once per tick.
            cur_nimg += batch_size
            done = (cur_nimg >= total_shapes * 1000)

            if (cur_tick != 0) and (cur_nimg < tick_start_nimg + cfg["resume"]["kimg_per_tick"] * 1000):
                continue

            # Print status line, accumulating the same information in training_stats.
            tick_end_time = time.time()
            fields = []
            fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
            fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / total_shapes):<9.2f}"]
            fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
            fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
            fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
            fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
            fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss *100 / psutil.virtual_memory().total):<6.2f}"]
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / (gpu.memoryTotal * 1024 * 1024)):<6.2f}"]
            fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / (gpu.memoryTotal * 1024 * 1024)):<6.2f}"]
            fields += [f"gpuload {training_stats.report0('Total gpu load', gpu.load*100):<6.2f}"]
            torch.cuda.reset_peak_memory_stats()
            dist.print0(' '.join(fields))

            # Check for abort.
            if dist.should_stop():
                dist.print0()
                dist.print0('Aborting...')

            # Save network snapshot.
            if (cfg["resume"]["snapshot_ticks"] is not None) and (cur_tick % cfg["resume"]["snapshot_ticks"] == 0):
                data = dict(ema=ema, loss_fn=loss_fn, dataset_name=dataset_obj.__str__())
                for key, value in data.items():
                    if isinstance(value, torch.nn.Module):
                        value = copy.deepcopy(value).eval().requires_grad_(False)
                        misc.check_ddp_consistency(value)
                        data[key] = value.cpu()
                    del value # conserve memory
                if dist.get_rank() == 0:
                    with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                        pickle.dump(data, f)
                del data # conserve memory

            # Save full dump of the training state.
            if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
                torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

            # Update logs.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
                stats_jsonl.flush()
            dist.update_progress(cur_nimg // 1000, total_shapes)

            # Update state.
            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time
            if done:
                break

    # Done.
    dist.print0()
    dist.print0('Exiting...')


def main(config):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    # Network options.

    # # Random seed.
    # if opts.seed is not None:
    #     c.seed = opts.seed
    # else:
    #     seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
    #     torch.distributed.broadcast(seed, src=0)
    #     c.seed = int(seed)

    # Transfer learning and resume.
    config["resume"]["resume_kimg"] = 0
    if config["resume"]["transfer"] is not None:
        config["resume"]["pkl"] = cfg["resume"]["transfer"]
        config["hyper_params"]["ema_rampup_ratio"] = None
    elif config["resume"]["pkl"] is not None:
        match = re.fullmatch(r'network-snapshot-(\d+).pkl', os.path.basename(config["resume"]["pkl"]))
        # if not match or not os.path.isfile(opts.resume):
        #     raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        # config["resume"]["pkl"] = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        config["resume"]["resume_kimg"] = int(match.group(1))
        #c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'uncond' #'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if config["perfs"]["fp16"] else 'fp32'
    desc = desc_from_config(config, dist.get_world_size())

    outdir = config["data"]["out"]
    # Pick output directory.
    if dist.get_rank() != 0:
        config["data"]["run"] = None
    # elif opts.nosubdir:
    elif 'id' in config["data"]:
        config["data"]["run"] = os.path.join(outdir, f'{config["data"]["id"]:05d}-{desc}')
        assert os.path.exists(config["data"]["run"])
    else:
        prev_run_dirs = []
        
        if os.path.isdir(outdir):
            prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        config["data"]["run"] = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
        print(config["data"]["run"], os.path.exists(config["data"]["run"]))
        assert not os.path.exists(config["data"]["run"])

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(config, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {config["data"]["run"]}')
    dist.print0(f'Dataset path:            {config["data"]["root_dir"]}')
    #dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {config["architecture"]}')
    #dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {config["hyper_params"]["batch_size"]}')
    dist.print0(f'Mixed-precision:         {config["perfs"]["fp16"]}')
    dist.print0()

    # Dry run?
    if config["misc"]["dry_run"]:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        run_dir = config["data"]["run"]
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
            json.dump(config, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop(config)

#----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch the training of spectral diffusion model.")
    parser.add_argument("--config", type=str, default="dfaust_fmap", help="Config file name")
    parser.add_argument("--device", type=str, default=0, help="Device ID")

    args = parser.parse_args()
    torch_device = f'cuda:{args.device}'
    cfg = yaml.safe_load(open(f"./config/diffusion/{args.config}.yaml", "r"))
    cfg['device'] = torch_device
    main(cfg)
