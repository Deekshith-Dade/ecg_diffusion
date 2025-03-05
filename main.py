from diffusion.cfg import GaussianDiffusion
from models.unet import Unet
from Training.trainer import Trainer
import dataset.DataTools as DataTools

import torch
import numpy as np
import random
import pandas as pd


def main():
    baseDir = '/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/kcl_explainability/explainability'
    dataDir = '/uu/sci.utah.edu/projects/ClinicalECGs/AllClinicalECGs/'
    
    num_classes = 2
    image_size = (8, 2500)
    channels = 1
    dim = 128
    batch_size = 12
    lr = 3e-4
    training_steps = 100000
    save_and_sample_every = 1000
    
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
    trainDataset = dataset_regular(
        baseDir = dataDir + 'pythonData/',
        ecgs = trainECGs['ECGFile'].tolist(),
        low_threshold= kclTaskParams['lowThresh'],
        high_threshold = kclTaskParams['highThresh'],
        kclVals=trainECGs['KCLVal'].tolist(),
        allowMismatchTime=False,
        randomCrop=True
    )
    
    model = Unet(
        dim = dim,
        num_classes=num_classes,
        cond_drop_prob=0.20,
        dim_mults=(1, 2, 4, 8, 16),
        channels=channels,
        attn_dim_head=32,
        attn_heads=4
    )
    
    # Print model summary to help with debugging
    print(f"Model initialized with image_size={image_size}, channels={channels}, dim={dim}")
    print(f"First layer expects input shape: (batch_size, {channels}, {image_size[0]}, {image_size[1]})")
    
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,
        sampling_timesteps=1000,
        objective='pred_noise',
        beta_schedule='cosine',
        ddim_sampling_eta=1.,
    )
    
    
    trainer = Trainer(
        diffusion_model=diffusion,
        train_batch_size=batch_size,
        train_lr = lr,
        train_num_steps=training_steps,
        ema_update_every=10,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=save_and_sample_every,
        num_samples=5,
        results_folder='./results/part6',
        dataset = trainDataset
    )
    
    trainer.train()

if __name__ == "__main__":
    main()

