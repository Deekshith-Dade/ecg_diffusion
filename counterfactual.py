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
    # Use a more recent successful model checkpoint
    path = "/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_counterfactual_diffusion/results/counter_2025-03-23_13-10-42/model-115.pt"
    print(f"Loading model from {path}")
    model, standardize_mean, standardize_std = load_unet_model(path)
    print("Model loaded successfully")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataDir = '/uu/sci.utah.edu/projects/ClinicalECGs/AllClinicalECGs/'
    
    num_classes = 2
    channels = 1
    image_size = (8, 2500)
    dim = 128
    batch_size = 8
    
    target_classes = torch.tensor([1] * batch_size, dtype=torch.long).to(device)
    
    # Use a more recent model checkpoint that produced good results during training
    results_folder = Path(f'./counterfactual_samples/test')
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
   
    # Step 1: Generate some random samples with class 1 (for comparison)
    print("Generating random samples with conditioning...")
    random_samples = diffusion.sample(target_classes, cond_scale=6.0).cpu()
    kv = dict(
                key = 'class',
                values = target_classes
            )
    plot_samples(random_samples, str(results_folder/f'random_sample_class1.png'), kv=kv)
    
    # Load test dataset for counterfactual generation
    print("Loading test dataset...")
    randSeed = 7777
    np.random.seed(randSeed)
    random.seed(randSeed)
    torch.manual_seed(randSeed)
    
    # Setup the test dataset
    timeCutoff = 900 #seconds
    lowerCutoff = 0 #seconds
    
    kclCohort = np.load(dataDir+'kclCohort_v1.npy',allow_pickle=True)
    data_types = {
        'DeltaTime': float,   
        'KCLVal': float,    
        'ECGFile': str,     
        'PatId': int,       
        'KCLTest': str      
    }
    kclCohort = pd.DataFrame(kclCohort,columns=['DeltaTime','KCLVal','ECGFile','PatId','KCLTest']) 
    for key in data_types.keys():
        kclCohort[key] = kclCohort[key].astype(data_types[key])

    kclCohort = kclCohort[kclCohort['DeltaTime']<=timeCutoff]
    kclCohort = kclCohort[kclCohort['DeltaTime']>lowerCutoff]

    kclCohort = kclCohort.dropna(subset=['DeltaTime']) 
    kclCohort = kclCohort.dropna(subset=['KCLVal']) 

    ix = kclCohort.groupby('ECGFile')['DeltaTime'].idxmin()
    kclCohort = kclCohort.loc[ix]

    numECGs = len(kclCohort)
    numPatients = len(np.unique(kclCohort['PatId']))
    
    print('Setting up train/val split')
    numTest = int(0.1 * numPatients)
    numTrain = numPatients - numTest
    assert (numPatients == numTrain + numTest), "Train/Test spilt incorrectly"
    RandomSeedSoAlswaysGetSameDatabseSplit = 1
    patientIds = list(np.unique(kclCohort['PatId']))
    random.Random(RandomSeedSoAlswaysGetSameDatabseSplit).shuffle(patientIds)
    
    trainPatientInds = patientIds[:numTrain]
    testPatientInds = patientIds[numTrain:numTest + numTrain]
    trainECGs = kclCohort[kclCohort['PatId'].isin(trainPatientInds)]
    testECGs = kclCohort[kclCohort['PatId'].isin(testPatientInds)]
    
    desiredTrainingAmount = len(trainECGs) // 1
    if desiredTrainingAmount != 'all':
        if len(trainECGs)>desiredTrainingAmount:
            trainECGs = trainECGs.sample(n=desiredTrainingAmount)
            
    kclTaskParams = dict(highThresh = 5, lowThresh=4, highThreshRestrict=8.5)
    trainECGs = trainECGs[(trainECGs['KCLVal']>=kclTaskParams['lowThresh']) & (trainECGs['KCLVal']<=kclTaskParams['highThreshRestrict'])]
    testECGs = testECGs[(testECGs['KCLVal']>=kclTaskParams['lowThresh']) & (testECGs['KCLVal']<=kclTaskParams['highThreshRestrict'])]
    
    dataset_regular = DataTools.ECG_KCL_Datasetloader
    testDataset = dataset_regular(
        baseDir = dataDir + 'pythonData/',
        ecgs = testECGs['ECGFile'].tolist(),
        low_threshold= kclTaskParams['lowThresh'],
        high_threshold = kclTaskParams['highThresh'],
        kclVals=testECGs['KCLVal'].tolist(),
        allowMismatchTime=False,
        randomCrop=True
    )
    
    # Collect a batch of class 0 samples
    print("Collecting class 0 samples for counterfactual generation...")
    class_0_samples = []
    class_0_indices = []

    for idx in range(len(testDataset)):
        sample = testDataset[idx]
        if sample['y'] == 0:  # Class 0
            class_0_samples.append(sample)
            class_0_indices.append(idx)
            if len(class_0_samples) >= batch_size:
                break

    # Check if we found enough class 0 samples
    if len(class_0_samples) < batch_size:
        print(f"Warning: Only found {len(class_0_samples)} samples of class 0, fewer than batch_size={batch_size}")
        # Adjust batch size if needed
        batch_size = len(class_0_samples)
        target_classes = target_classes[:batch_size]

    # Create batch from collected samples
    source_images = torch.stack([sample['image'] for sample in class_0_samples[:batch_size]]).to(device)
    source_classes = torch.tensor([sample['y'] for sample in class_0_samples[:batch_size]], dtype=torch.long).to(device)
    
    print(f"Created batch with {source_images.shape[0]} samples of class 0")
    kv = dict(
                key = 'class',
                values = source_classes
            )
    # Save the original images for comparison
    plot_samples(source_images.cpu(), str(results_folder/f'original_class0.png'), kv=kv)
    
    # Step 2: Generate counterfactuals (convert class 0 to class 1)
    print("Starting counterfactual generation process...")
    
    # First, convert the clean images to noisy versions 
    sampling_ratio = 0.25  # Changed to match training configuration (25% instead of 75%)
    print("Converting clean images to noisy versions...")
    noisy_images, noise_progression = diffusion.ddim_counterfactual_sample_from_clean_to_noisy(
        source_images, 
        sampling_ratio=sampling_ratio,
        progress=True
    )
    print("Done converting to noise")
    kv = dict(
                key = 'class',
                values = source_classes
            )
    # Save the noisy intermediate result
    plot_samples(noisy_images.cpu(), str(results_folder/f'noisy_intermediate.png'), kv=kv)
    
    # Now generate counterfactuals by conditioning on the target class
    print("Generating counterfactuals from noisy images...")
    counterfactual_images, diffusion_progression = diffusion.ddim_counterfactual_sample_from_noisy_to_counterfactual(
        noisy_images, 
        target_classes, 
        sampling_ratio=sampling_ratio, 
        cond_scale=6.0,
        progress=True
    )
    print("Done generating counterfactuals")
    kv = dict(
                key = 'class',
                values = target_classes
            )
    # Save the counterfactual results
    plot_samples(counterfactual_images.cpu(), str(results_folder/f'counterfactual_class1.png'), kv=kv)
    
   
    
    print(f"All results saved to {results_folder}")
    
if __name__ == "__main__":
    main()