import pickle 
import time
import pandas as pd
import numpy as np
import random

import dataset.DataTools as DataTools
import torch
import os



def splitKCLPatients(seed):
    start = time.time()
    dataDir = '/uu/sci.utah.edu/projects/ClinicalECGs/AllClinicalECGs/'

    timeCutoff = 900 #seconds
    lowerCutoff = 0 #seconds

    print("Finding Patients")
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
    #remove those above cutoff
    kclCohort = kclCohort[kclCohort['DeltaTime']<=timeCutoff]
    kclCohort = kclCohort[kclCohort['DeltaTime']>lowerCutoff]#remove those below lower cutoff
    #remove nans
    kclCohort = kclCohort.dropna(subset=['DeltaTime']) 
    kclCohort = kclCohort.dropna(subset=['KCLVal']) 
    #kclCohort = kclCohort[0:2000]#jsut for testing
    #for each ECG, keep only the ECG-KCL pair witht he shortest delta time
    ix = kclCohort.groupby('ECGFile')['DeltaTime'].idxmin()
    kclCohort = kclCohort.loc[ix]

    numPatients = len(np.unique(kclCohort['PatId']))

    print('setting up train/val split')
    numTest = int(0.1 * numPatients)
    numTrain = numPatients - numTest
    assert (numPatients == numTrain + numTest), "Train/Test spilt incorrectly"
    patientIds = list(np.unique(kclCohort['PatId']))
    random.Random(seed).shuffle(patientIds)

    trainPatientInds = patientIds[:numTrain]
    testPatientInds = patientIds[numTrain:numTest + numTrain]
    trainECGs = kclCohort[kclCohort['PatId'].isin(trainPatientInds)]
    testECGs = kclCohort[kclCohort['PatId'].isin(testPatientInds)]

    perNetLossParams = dict(learningRate = 1e-3,highThresh = 5, lowThresh=4 ,type = 'binary cross entropy')

    
    trainECGs_normal = trainECGs[(trainECGs['KCLVal']>=perNetLossParams['lowThresh']) & (trainECGs['KCLVal']<=perNetLossParams['highThresh'])]
    trainECGs_abnormal = trainECGs[(trainECGs['KCLVal']<perNetLossParams['lowThresh']) | (trainECGs['KCLVal']>perNetLossParams['highThresh'])]

    #any additional processing
    #for this trial, only normal vs high. Remove the lows
    trainECGs_abnormal = trainECGs_abnormal[trainECGs_abnormal['KCLVal']>perNetLossParams['lowThresh']]
    testECGs = testECGs[testECGs['KCLVal']>perNetLossParams['lowThresh']]
    
    
    os.makedirs('kcl_patient_splits', exist_ok=True)

    with open('kcl_patient_splits/train_normal_patients.pkl', 'wb') as file:
        pickle.dump(trainECGs_normal, file)
    
    with open('kcl_patient_splits/train_abnormal_patients.pkl', 'wb') as file:
        pickle.dump(trainECGs_abnormal, file)
    
    with open('kcl_patient_splits/test_patients.pkl', 'wb') as file:
        pickle.dump(testECGs, file)

    print(f'Found {len(testECGs)} tests and {len(trainECGs)} trains. In training, {len(trainECGs_normal)} are normal, {len(trainECGs_abnormal)} are abnormal')
    print(f'The process took {time.time()-start} seconds')


def dataprepKCL(args):
    dataDir = '/uu/sci.utah.edu/projects/ClinicalECGs/AllClinicalECGs/'
    normEcgs = False

    with open('kcl_patient_splits/test_patients.pkl','rb') as f:
        testECGs = pickle.load(f)
    with open('kcl_patient_splits/train_normal_patients.pkl','rb') as f:
        trainECGs_normal = pickle.load(f)
    with open('kcl_patient_splits/train_abnormal_patients.pkl','rb') as f:
        trainECGs_abnormal = pickle.load(f)

    trainECGs_normal_count = len(trainECGs_normal)
    trainECGs_abnormal_count = len(trainECGs_abnormal)

    finetuning_ratios = args.finetuning_ratios
    num_finetuning = [[int(trainECGs_normal_count * ratio), int(trainECGs_abnormal_count * ratio)] for ratio in finetuning_ratios]

    train_loaders = []
    dataset_lengths = []
    for i in num_finetuning:
        finetuning_patients_normal = trainECGs_normal.sample(n=i[0])
        finetuning_patients_abnormal = trainECGs_abnormal.sample(n=i[1])

        trainData_normal_dataset = DataTools.ECG_KCL_Datasetloader_Classify(baseDir=dataDir+'pythonData/', 
                                                                    ecgs=finetuning_patients_normal['ECGFile'].tolist(),
                                                                    kclVals=finetuning_patients_normal['KCLVal'].tolist(),
                                                                    normalize=normEcgs,
                                                                    allowMismatchTime=False,
                                                                    randomCrop=True)
        trainData_abnormal_dataset = DataTools.ECG_KCL_Datasetloader_Classify(baseDir=dataDir+'pythonData/', 
                                                                    ecgs=finetuning_patients_abnormal['ECGFile'].tolist(),
                                                                    kclVals=finetuning_patients_abnormal['KCLVal'].tolist(),
                                                                    normalize=normEcgs,
                                                                    allowMismatchTime=False,
                                                                    randomCrop=True)

        trainData_normal_loader = torch.utils.data.DataLoader(trainData_normal_dataset,shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
        trainData_abnormal_loader = torch.utils.data.DataLoader(trainData_abnormal_dataset,shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

        dataset_lengths.append([len(trainData_normal_dataset), len(trainData_abnormal_dataset)])
        train_loaders.append([trainData_normal_loader, trainData_abnormal_loader])
    
    test_dataset = DataTools.ECG_KCL_Datasetloader_Classify(baseDir=dataDir+'pythonData/',
                                                   ecgs=testECGs['ECGFile'].tolist(),
                                                   kclVals=testECGs['KCLVal'].tolist(),
                                                   normalize=normEcgs,
                                                   allowMismatchTime=False,
                                                   randomCrop=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Preparing Finetuning with {dataset_lengths} number of ECGs and validation with {len(test_dataset)} number of ECGs")

    return train_loaders, test_loader
  