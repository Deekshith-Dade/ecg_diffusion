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
    
    checkpoint = torch.load(path, weights_only=False)
    
    num_classes = 2
    channels = 1
    
    model = Unet(
        dim= 128,
        channels=channels,
        dim_mults=(1, 2, 4, 8, 16),
        num_classes=num_classes,
        cond_drop_prob=0.35,
        )
    
    checkpoint_state_dict = checkpoint["model"]
    new_state_dict = {k.replace('module.model.', ''): v for k, v in checkpoint_state_dict.items()}
    checkpoint["model"] = new_state_dict
    model.load_state_dict(checkpoint["model"], strict=False)
    return model, checkpoint


def main():
    path = "/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_counterfactual_diffusion/results/part7/model-1.pt"
    model, checkpoint = load_unet_model(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataDir = '/uu/sci.utah.edu/projects/ClinicalECGs/AllClinicalECGs/'
    
    num_classes = 2
    channels = 1
    image_size = (8, 2500)
    dim = 128
    batch_size = 8
    
    classes = torch.tensor([1] * batch_size, dtype=torch.long)
    classes = classes.to(device)
    
    results_folder = Path(f'./counterfactual_samples/sample1')
    results_folder.mkdir(parents=True, exist_ok=True)
    
    
    
    diffusion = GaussianDiffusion(
        model=model,
        image_size=image_size,
        timesteps = 1000,
        sampling_timesteps=100,
        beta_schedule=getattr(checkpoint, 'beta_schedule', 'cosine'),
        ddim_sampling_eta=1.0,
        standardize_mean=checkpoint['model']['module.standardize_mean'],
        standardize_std=checkpoint['model']['module.standardize_std']
    ).to(device)
    
   
    
    output = diffusion.sample(classes, cond_scale=5.).cpu()
    plot_samples(output, str(results_folder/f'sample.png'))
    
    import pdb; pdb.set_trace()
    
    randSeed = 7777
    np.random.seed(randSeed)
    
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
    
    print('setting up train/val split')
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

    # Create batch from collected samples
    images = torch.stack([sample['image'] for sample in class_0_samples[:batch_size]]).to(device)
    actual_classes = torch.tensor([sample['y'] for sample in class_0_samples[:batch_size]], dtype=torch.long).to(device)

    print(f"Created batch with {images.shape[0]} samples of class 0")
    
    sampling_ratio = 0.75
    exogenous_noise, abduction_progression = diffusion.ddim_counterfactual_sample_from_clean_to_noisy(images, sampling_ratio=sampling_ratio)
    print("Done getting the noise")
    
    init_image = exogenous_noise
    cond_scale = 6.
    
    counterfactual_image, diffusion_progression = diffusion.ddim_counterfactual_sample_from_noisy_to_counterfactual(init_image, classes, sampling_ratio, cond_scale=cond_scale)
    print("Done getting the counterfactuals")
    import pdb; pdb.set_trace()
    
    
    
    

if __name__ == "__main__":
    main()