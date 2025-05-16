import pickle 
import time
import pandas as pd
import numpy as np
import random

import dataset.DataTools as DataTools
import torch
import os


def splitPatientsLVEF(dataDir, seed):
    start = time.time()
    normEcgs = False

    # Loading Data
    print('Finding Patients')
    allData = DataTools.ECG_LVEF_DatasetLoader(baseDir=dataDir, normalize=normEcgs)
    patientIds = np.array(allData.patients)
    numPatients = patientIds.shape[0]

    # Data
    train_split = 0.9
    num_train = int(train_split * numPatients)
    num_validation = numPatients - num_train

    patientInds = list(range(numPatients))
    random.Random(seed).shuffle(patientInds)

    pre_train_patient_indices = patientInds[:num_train]
    validation_patient_indices = patientInds[num_train:num_train + num_validation]

    pre_train_patients = patientIds[pre_train_patient_indices].squeeze()
    validation_patients = patientIds[validation_patient_indices].squeeze()

    with open('lvef_patient_splits/pre_train_patients.pkl', 'wb') as file:
        pickle.dump(pre_train_patients, file)
    with open('lvef_patient_splits/validation_patients.pkl', 'wb') as file:
        pickle.dump(validation_patients, file)
    print(f"Out of Total {numPatients} Splitting {len(pre_train_patients)} for Training, {len(validation_patients)} for validation")
    print(f'The process took {time.time()-start} seconds')

def dataprepLVEF(dataDir):
    normEcgs = False

    print("Preparing Data For Finetuning")
    with open('lvef_patient_splits/validation_patients.pkl', 'rb') as file:
        validation_patients = pickle.load(file)
    
    with open('lvef_patient_splits/pre_train_patients.pkl', 'rb') as file:
        pre_train_patients = pickle.load(file)
    
    # num_train_patients = len(pre_train_patients)
    

    # patientInds = list(range(num_train_patients))
    # random.shuffle(patientInds)
    
    # finetuning_patients = pre_train_patients[patientInds]

    train_dataset = DataTools.ECG_LVEF_DatasetLoader(baseDir=dataDir, patients=pre_train_patients.tolist(), normalize=normEcgs)
    
    test_dataset = DataTools.ECG_LVEF_DatasetLoader(baseDir=dataDir, patients=validation_patients.tolist(), normalize=normEcgs)
    

    print(f"Preparing Training with {len(train_dataset)} number of ECGs and validation with {len(test_dataset)} number of ECGs")

    return train_dataset, test_dataset




def getKCLTrainTestDataset(scale_training_size, dataDir):
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
    
    desiredTrainingAmount = len(trainECGs) // scale_training_size
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
    print(f'Number of Training Examples: {len(trainDataset)}')
    
    testDataset = dataset_regular(
        baseDir = dataDir + 'pythonData/',
        ecgs = testECGs['ECGFile'].tolist(),
        low_threshold= kclTaskParams['lowThresh'],
        high_threshold = kclTaskParams['highThresh'],
        kclVals=testECGs['KCLVal'].tolist(),
        allowMismatchTime=False,
        randomCrop=True
    )

    return trainDataset, testDataset, kclTaskParams, timeCutoff, lowerCutoff, randSeed

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

    # print("Saving abnormal test samples...")
    # abnormal_samples = []
    # abnormal_count = 0

    # # Iterate through the test dataset directly
    # abnormal_count = 0
    # abnormal_ecgs = []
    # abnormal_kcl_vals = []
    # abnormal_paths = []
    
    # for i in range(len(testDataset)):
    #     try:
    #         item = testDataset[i]
    #         # Check if it's an abnormal sample (y = 0)
    #         if item['y'] == 0:
    #             abnormal_ecgs.append(item['image'])
    #             abnormal_kcl_vals.append(item['kclVal'].item())  # Store the KCL value
    #             abnormal_paths.append(item['ecgPath'])  # Store the ECG path
    #             abnormal_count += 1
                
    #             # Print progress
    #             if abnormal_count % 10 == 0:
    #                 print(f"Found {abnormal_count} abnormal samples so far")
                
    #             # # Stop after collecting 100 samples
    #             # if abnormal_count >= 100:
    #             #     break
    #     except Exception as e:
    #         print(f"Error processing sample {i}: {e}")
    #         continue

    # # Check if we found enough samples
    # if abnormal_count < 100:
    #     print(f"Warning: Only found {abnormal_count} abnormal samples in the test dataset")
    # else:
    #     print(f"Successfully collected {abnormal_count} abnormal samples")

    # # Save the abnormal samples, KCL values, and paths in a single file
    # if abnormal_ecgs:
    #     abnormal_tensor = torch.stack(abnormal_ecgs)
    #     abnormal_kcl_tensor = torch.tensor(abnormal_kcl_vals)
        
    #     # Save as a single dictionary in a file
    #     save_data = {
    #         'ECGs': abnormal_tensor,
    #         'KCLs': abnormal_kcl_tensor,
    #         'Paths': abnormal_paths
    #     }
        
    #     torch.save(save_data, f"{baseDir}/abnormal_data.pt")
    #     print(f"Saved abnormal ECGs, KCL values, and paths to {baseDir}/abnormal_data.pt")