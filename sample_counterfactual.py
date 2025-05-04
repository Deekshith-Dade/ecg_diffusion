from networkx import generate_adjlist
import torch
from models.unet import Unet
import numpy as np
import random
import pandas as pd
import dataset.DataTools as DataTools
from diffusion.cfg import GaussianDiffusion

from utils.plot_utils import plot_samples

from pathlib import Path
import matplotlib.pyplot as plt



def load_unet_model(path):
    
    checkpoint = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
    
    num_classes = 2
    channels = 1
    
    model = Unet(
        dim=128,
        channels=channels,
        dim_mults=(1, 2, 4, 8, 16),
        num_classes=num_classes,
        cond_drop_prob=0.35,
        attn_dim_head=64,
        attn_heads=8
        )
    
    
    checkpoint_state_dict = checkpoint.get("model")
    state_dict = {key.replace('module.model.', ''): value for key, value in checkpoint_state_dict.items() if 'module.model' in key}

    model.load_state_dict(state_dict, strict=True)
    
    standardize_mean = checkpoint.get('mean')
    standardize_std = checkpoint.get('std')
    
    if standardize_mean is None or standardize_std is None:
        print("Warning: Mean or std not found in checkpoint, using defaults")
    else:
        print(f"Loaded standardization parameters - Mean: {standardize_mean}")
        print(f"Loaded standardization parameters - Std: {standardize_std}")
    
    return model, standardize_mean, standardize_std

def main():
    path = "/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_counterfactual_diffusion/results/counter_2025-03-13_22-26-57/model-230.0.pt"
    print(f"Loading model from {path}")
    model, standardize_mean, standardize_std = load_unet_model(path)
    print("Model loaded successfully")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_size = (8, 2500)
    batch_size = 8
    target_classes = torch.tensor([1] * batch_size, dtype=torch.long).to(device)
    folder = "currentAbnormalData2"
    # Use a more recent model checkpoint that produced good results during training
    results_folder = Path(f'./{folder}/plots')
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # Create diffusion model with proper standardization parameters
    print("Creating diffusion model")
    diffusion = GaussianDiffusion(
        model=model,
        image_size=image_size,
        timesteps=1000,
        sampling_timesteps=1000,  # Increased from 100 for better quality
        objective='pred_noise',
        beta_schedule='cosine',  # Use the beta schedule from checkpoint if available
        ddim_sampling_eta=1.0,
    ).to(device)
    standardize_mean = standardize_mean.to(device)
    standardize_std = standardize_std.to(device)
    diffusion._set_mean_std(standardize_mean, standardize_std)
    print("Diffusion model created successfully")
    
    generate_random_samples = True
    if generate_random_samples:
        # Step 1: Generate some random samples with class 1 (for comparison)
        print("Generating random samples with conditioning...")
        random_samples = diffusion.sample(target_classes, cond_scale=6.0).cpu()
        kv = dict(
                    key = 'class',
                    values = target_classes
                )
        plot_samples(random_samples, str(results_folder/f'random_sample_class1.png'), kv=kv)

    
    # Step 2: Load the ECG data
    path = f"./{folder}/abnormal_data.pt"
    data = torch.load(path)
    
    ecgs = data['ECGs'] 
    kcls = data['KCLs']
    paths = data['Paths']
    
    # Add channel dimension if not present - reshape from (N, 8, 2500) to (N, 1, 8, 2500)
    if len(ecgs.shape) == 3:  # Shape is (N, 8, 2500)
        print(f"Original ECG shape: {ecgs.shape}")
        ecgs = np.expand_dims(ecgs, axis=1)
        print(f"Reshaped ECG with channel dimension: {ecgs.shape}")
    
    # Set up for batch processing
    batch_size = 50
    num_samples = ecgs.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    # Set target classes to 0 (healthy)
    target_class = 1
    
    # Store all counterfactual results
    all_counterfactuals = []
    
    print(f"Starting counterfactual generation for {num_samples} ECGs in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        current_batch_size = end_idx - start_idx
        
        print(f"Processing batch {batch_idx+1}/{num_batches}, samples {start_idx} to {end_idx-1}")
        
        # Get current batch
        source_images = ecgs[start_idx:end_idx].to(device)
        
        # Create target classes tensor for current batch
        target_classes = torch.tensor([target_class] * current_batch_size, dtype=torch.long).to(device)
        
        # For visualization, save some original samples
        if batch_idx == 0:
            kv = dict(key='class', values=torch.ones(current_batch_size, dtype=torch.long))
            plot_samples(source_images.cpu(), str(results_folder/f'original_abnormal_samples.png'), kv=kv)
        
        # Generate counterfactuals (convert abnormal to healthy)
        print("Converting abnormal ECGs to healthy versions...")
        
        # First, convert the clean images to noisy versions 
        sampling_ratio = 0.75  # Same as in counterfactual.py
        noisy_images, noise_progression = diffusion.ddim_counterfactual_sample_from_clean_to_noisy(
            source_images, 
            sampling_ratio=sampling_ratio,
            progress=True
        )
        
        # Now generate counterfactuals by conditioning on the target class
        counterfactual_images, diffusion_progression = diffusion.ddim_counterfactual_sample_from_noisy_to_counterfactual(
            noisy_images, 
            target_classes, 
            sampling_ratio=sampling_ratio, 
            cond_scale=6.0,
            progress=True
        )
        
        # Save sample visualizations for first batch
        if batch_idx == 0:
            kv = dict(key='class', values=target_classes.cpu())
            plot_samples(counterfactual_images.cpu(), str(results_folder/f'counterfactual_healthy_samples.png'), kv=kv)
        
        # Store the counterfactual results
        all_counterfactuals.append(counterfactual_images.cpu().numpy())
    
    # Combine all batches
    all_counterfactuals = np.concatenate(all_counterfactuals, axis=0)
    
    # Remove channel dimension before saving if necessary
    if all_counterfactuals.shape[1] == 1:  # Shape is (N, 1, 8, 2500)
        all_counterfactuals = all_counterfactuals[:, 0, :, :]  # Convert to (N, 8, 2500)
        print(f"Removed channel dimension before saving. Final shape: {all_counterfactuals.shape}")
    
    # Save as .pt file
    print(f"Saving {len(all_counterfactuals)} counterfactual ECGs to modified.pt")
    torch.save({
        'ECGs': all_counterfactuals,
        'KCLs': kcls, 
        'paths': paths
    }, f'./{folder}/modified.pt')
    
    print(f"Conversion complete. Results saved to ./{folder}/modified.pt")

if __name__ == "__main__":
    main()


