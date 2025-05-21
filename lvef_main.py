import test
from diffusion.cfg import GaussianDiffusion
from models.unet import Unet
from Training.trainer import Trainer
import dataset.DataTools as DataTools
from dataset.utils import dataprepLVEF

import torch
import numpy as np
import random
import pandas as pd
import wandb
import datetime
import json

from dataset.utils import dataprepLVEF, getKCLTrainTestDataset, splitPatientsLVEF

def main():
    dataDir = '/uu/sci.utah.edu/projects/ClinicalECGs/LVEFCohort/pythonData/'
    
    logtowandb = True
    
    num_classes = 2
    image_size = (8, 2500)
    channels = 1
    dim = 128
    batch_size = 10 * torch.cuda.device_count()
    lr = 1e-4
    training_steps = 100000
    save_and_sample_every = 1000
    scale_training_size = 1
    randSeed = 7777
    splitPatientsLVEF(dataDir, randSeed)
    trainDataset, testDataset = dataprepLVEF(dataDir)
    print(f'Number of Training Examples: {len(trainDataset)}')

    unetParams = dict(
                cond_drop_prob=0.5,
                dim_mults = (1, 2, 4, 8, 16),
                attn_dim_heads = 64,
                attn_heads = 8
            )
    
    
    model = Unet(
        dim = dim,
        num_classes=num_classes,
        cond_drop_prob=unetParams['cond_drop_prob'],
        dim_mults=unetParams['dim_mults'],
        channels=channels,
        attn_dim_head=unetParams['attn_dim_heads'],
        attn_heads=unetParams['attn_heads']
    )
    
    # Print model summary to help with debugging
    print(f"Model initialized with image_size={image_size}, channels={channels}, dim={dim}")
    print(f"First layer expects input shape: (batch_size, {channels}, {image_size[0]}, {image_size[1]})")
    
    diffusionParams = dict(
        timesteps = 1000,
        sampling_timesteps = 1000,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 1.
    )
    
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=diffusionParams['timesteps'],
        sampling_timesteps=diffusionParams['sampling_timesteps'],
        objective=diffusionParams['objective'],
        beta_schedule=diffusionParams['beta_schedule'],
        ddim_sampling_eta=diffusionParams['ddim_sampling_eta'],
    )
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    trainerParams = dict(
        ema_update_every = 10,
        adam_betas = (0.9, 0.99),
        num_samples = 5,
        results_folder = f'./lvef_results/counter_{formatted_time}',
        sampling_cond_scale = 10.,
        counterfactual_sampling_ratio = 0.5,
        counterfactual_sampling_cond_scale = 10.
    )
    
    config = dict(
        classes = num_classes,
        image_size = image_size,
        channels = channels,
        dim = dim,
        batch_size = batch_size,
        lr = lr,
        training_steps = training_steps,
        save_and_sample_every=save_and_sample_every,
        scale_training_size = scale_training_size,
        randSeed = randSeed,
        unetParams = unetParams,
        diffusionParams = diffusionParams,
        
    )
    
    if logtowandb:
        wandbrun = wandb.init(
            project="CounterfactualDiff_LVEF",
            notes=f"",
            tags=["training","KCL"],
            config=config,
            entity="deekshith",
            reinit=True,
            name=f"{"counterfactual"}_{datetime.datetime.now()}",
        )
    
    trainer = Trainer(
        diffusion_model=diffusion,
        train_batch_size=batch_size,
        train_lr = lr,
        train_num_steps=training_steps,
        ema_update_every=trainerParams['ema_update_every'],
        adam_betas=trainerParams['adam_betas'],
        save_and_sample_every=save_and_sample_every,
        num_samples=trainerParams['num_samples'],
        results_folder=trainerParams['results_folder'],
        dataset = trainDataset,
        test_dataset = testDataset,
        sampling_cond_scale=trainerParams['sampling_cond_scale'],
        counterfactual_sampling_ratio=trainerParams['counterfactual_sampling_ratio'],
        counterfactual_sampling_cond_scale=trainerParams['counterfactual_sampling_cond_scale'],
        logtowandb = logtowandb
    )
    
    
    
    trainer.train()

if __name__ == "__main__":
    main()

