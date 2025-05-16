import numpy as np
import torch
from torch.utils.data import Dataset
import os
import torch.nn.functional as F
import json
# import Loader
import torchvision.transforms as transforms

class DataLoaderError(Exception):
    pass

class ECG_KCL_Datasetloader_Classify(Dataset):
	def __init__(self,baseDir='',ecgs=[],kclVals=[],normalize =True, 
				 normMethod='0to1',rhythmType='Rhythm',allowMismatchTime=True,
				 mismatchFix='Pad',randomCrop=False,cropSize=2500,expectedTime=5000):
		self.baseDir = baseDir
		self.rhythmType = rhythmType
		self.normalize = normalize
		self.normMethod = normMethod
		self.ecgs = ecgs
		self.kclVals = kclVals
		self.expectedTime = expectedTime
		self.allowMismatchTime = allowMismatchTime
		self.mismatchFix = mismatchFix
		self.randomCrop = randomCrop
		self.cropSize = cropSize
		if self.randomCrop:
			self.expectedTime = self.cropSize

	def __getitem__(self,item):
		ecgName = self.ecgs[item].replace('.xml',f'_{self.rhythmType}.npy')
		ecgPath = os.path.join(self.baseDir,ecgName)
		ecgData = np.load(ecgPath)

		kclVal = torch.tensor(self.kclVals[item])
		ecgs = torch.tensor(ecgData).float() #unsqueeze it to give it one channel\

		if self.randomCrop:
			startIx = 0
			if ecgs.shape[-1]-self.cropSize > 0:
				startIx = torch.randint(ecgs.shape[-1]-self.cropSize,(1,))
			ecgs = ecgs[...,startIx:startIx+self.cropSize]

		if ecgs.shape[-1] != self.expectedTime:
			if self.allowMismatchTime:
				if self.mismatchFix == 'Pad':
					ecgs=F.pad(ecgs,(0,self.expectedTime-ecgs.shape[-1]))
				if self.mismatchFix == 'Repeat':
					timeDiff = self.expectedTime - ecgs.shape[-1]
					ecgs=torch.cat((ecgs,ecgs[...,0:timeDiff]))

			else:
				raise DataLoaderError('You are not allowed to have mismatching data lengths.')

		if self.normalize:
			if self.normMethod == '0to1':
				if not torch.allclose(ecgs,torch.zeros_like(ecgs)):
					ecgs = ecgs - torch.min(ecgs)
					ecgs = ecgs / torch.max(ecgs)
				else:
					print(f'All zero data for item {item}, {ecgPath}')
			
		if torch.any(torch.isnan(ecgs)):
			print(f'Nans in the data for item {item}, {ecgPath}')
			raise DataLoaderError('Nans in data')
		return ecgs, kclVal

	def __len__(self):
		return len(self.ecgs)

class ECG_KCL_Datasetloader(Dataset):
    def __init__(self, baseDir='', ecgs=[], kclVals=[], low_threshold=4.0, high_threshold=5.0, rhythmType='Rhythm',
                 allowMismatchTime=True, mismatchFix='Pad', randomCrop=False,
                 cropSize=2500, expectedTime=5000):
        self.baseDir = baseDir
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.rhythmType = rhythmType
        self.ecgs = ecgs
        self.kclVals = kclVals
        self.expectedTime = expectedTime,
        self.allowMismatchTime = allowMismatchTime
        self.mismatchFix = mismatchFix
        self.cropSize = cropSize
        self.randomCrop = randomCrop
        if self.randomCrop:
            self.expectedTime = self.cropSize
    
    def __getitem__(self, item):
        ecgName = self.ecgs[item].replace('.xml', f'_{self.rhythmType}.npy')
        ecgPath = os.path.join(self.baseDir, ecgName)
        ecgData = np.load(ecgPath)
        
        kclVal = torch.tensor(self.kclVals[item])
        ecgs = torch.tensor(ecgData).unsqueeze(0).float()
        # temp_ecgs = ecgs.clone()
        # ecgs[0, 0:2] = temp_ecgs[0, 2:4]
        # ecgs[0, 2:4] = temp_ecgs[0, 0:2]
        
        if self.randomCrop:
            startIx = 0
            if ecgs.shape[-1]-self.cropSize > 0:
                startIx = torch.randint(ecgs.shape[-1] - self.cropSize, (1,))
            ecgs = ecgs[...,startIx:startIx+self.cropSize]
        
        if ecgs.shape[-1] != self.expectedTime:
            if self.allowMismatchTime:
                if self.mismatchFix == 'Pad':
                    ecgs = F.pad(ecgs, (0, self.expectedTime-ecgs.shape[-1]))
                if self.mismatchFix == 'Repeat':
                    timeDiff = self.expectedTime - ecgs.shape[-1]
                    ecgs = torch.cat((ecgs, ecgs[...,0:timeDiff]))
            else:
                raise DataLoaderError('You are not allowed to have mismatching data lengths.')
        
        if torch.any(torch.isnan(ecgs)):
            print(f'Nans in the data for item {item}, {ecgPath}')
            raise DataLoaderError('Nans in data')

        item = {}
        item['image'] = ecgs
        item['y'] = 1 if kclVal <= self.high_threshold and kclVal >= self.low_threshold else 0
        item['key'] = 'kclVal'
        item['val'] = kclVal
        item['ecgPath'] = ecgPath
        return item
    
    def __len__(self):
        return len(self.ecgs)


class ECG_KCL_Augs_Datasetloader(Dataset):
	def __init__(self,baseDir='',ecgs=[],kclVals=[],normalize =True, 
				 normMethod='0to1',rhythmType='Rhythm',allowMismatchTime=True,
				 mismatchFix='Pad',randomCrop=False,cropSize=2500,expectedTime=5000):
		self.baseDir = baseDir
		self.rhythmType = rhythmType
		augmentation = [
			Loader.SpatialTransform(),
		]
		self.augs = Loader.TwoCropsTransform(transforms.Compose(augmentation))
		self.normalize = normalize
		self.normMethod = normMethod
		self.ecgs = ecgs
		self.kclVals = kclVals
		self.expectedTime = expectedTime
		self.allowMismatchTime = allowMismatchTime
		self.mismatchFix = mismatchFix
		self.randomCrop = randomCrop
		self.ecgs = [ecg for ecg in self.ecgs for _ in range(2)]
		self.kclVals = [kcl for kcl in self.kclVals for _ in range(2)]
		self.augmentationFlag = [False, True] * len(self.kclVals)
		self.cropSize = cropSize
		if self.randomCrop:
			self.expectedTime = self.cropSize
   		
	

	def __getitem__(self,item):
		ecgName = self.ecgs[item].replace('.xml',f'_{self.rhythmType}.npy')
		ecgPath = os.path.join(self.baseDir,ecgName)
		ecgData = np.load(ecgPath)

		kclVal = torch.tensor(self.kclVals[item])
		ecgs = torch.tensor(ecgData).unsqueeze(0).float() #unsqueeze it to give it one channel\

		if self.randomCrop:
			startIx = 0
			if ecgs.shape[-1]-self.cropSize > 0:
				startIx = torch.randint(ecgs.shape[-1]-self.cropSize,(1,))
			ecgs = ecgs[...,startIx:startIx+self.cropSize]

		if ecgs.shape[-1] != self.expectedTime:
			if self.allowMismatchTime:
				if self.mismatchFix == 'Pad':
					ecgs=F.pad(ecgs,(0,self.expectedTime-ecgs.shape[-1]))
				if self.mismatchFix == 'Repeat':
					timeDiff = self.expectedTime - ecgs.shape[-1]
					ecgs=torch.cat((ecgs,ecgs[...,0:timeDiff]))

			else:
				raise DataLoaderError('You are not allowed to have mismatching data lengths.')

		if self.normalize:
			if self.normMethod == '0to1':
				if not torch.allclose(ecgs,torch.zeros_like(ecgs)):
					ecgs = ecgs - torch.min(ecgs)
					ecgs = ecgs / torch.max(ecgs)
				else:
					print(f'All zero data for item {item}, {ecgPath}')
     
		if self.augmentationFlag[item]:
			ecgs = self.augs(ecgs)[0]
   
		if torch.any(torch.isnan(ecgs)):
			print(f'Nans in the data for item {item}, {ecgPath}')
			raise DataLoaderError('Nans in data')
		return ecgs, kclVal

	def __len__(self):
		return len(self.ecgs)

class ECG_LVEF_DatasetLoader(Dataset):
    
    def __init__(self, baseDir='', patients=[], normalize=True, normMethod='unitrange', rhythmType='Rhythm', numECGstoFind=1):
        self.baseDir = baseDir
        self.rhythmType = rhythmType
        self.normalize = normalize
        self.normMethod = normMethod
        self.fileList = []
        self.patientLookup = []

        if len(patients) == 0:
            self.patients = os.listdir(baseDir)
        else:
            self.patients = patients
        
        if type(self.patients[0]) is not str:
            self.patients = [str(pat) for pat in self.patients]
        
        if numECGstoFind == 'all':
            for pat in self.patients:
                self.findEcgs(pat, 'all')
        else:
            for pat in self.patients:
                self.findEcgs(pat, numECGstoFind)
    
    def findEcgs(self, patient, numberToFind=1):
        patientInfoPath = os.path.join(self.baseDir, patient, 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        numberOfEcgs = patientInfo['numberOfECGs']

        if(numberToFind == 1) | (numberOfEcgs == 1):
            for i in range(2):
                ecgId = str(patientInfo["ecgFileIds"][0])
                zeros = 5 - len(ecgId)
                ecgId = "0"*zeros+ ecgId
                self.fileList.append(os.path.join(patient,
                                    f'ecg_0',
                                    f'{ecgId}_{self.rhythmType}.npy'))
                self.patientLookup.append(f"{patient}_{i}")
        else:
            for ecgIx in range(numberOfEcgs):
                for i in range(2):
                    self.fileList.append(os.path.join(patient,
                                                f'ecg_{ecgIx}',
                                                f'{patientInfo["ecgFields"][ecgIx]}_{self.rhythmType}.npy'))
                    self.patientLookup.append(f"{patient}_{i}")
        
    
    def __getitem__(self, item):
        patient = self.patientLookup[item][:-2]
        segment = self.patientLookup[item][-1]

        patientInfoPath = os.path.join(self.baseDir, patient, 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        
        ecgPath = os.path.join(self.baseDir,
                               self.fileList[item])
        
        ecgData = np.load(ecgPath)
        if(segment == '0'):
            ecgData = ecgData[:, 0:2500]
        else:
            ecgData = ecgData[:, 2500:]

        ejectionFraction = torch.tensor(patientInfo['ejectionFraction'])
        ecgs = torch.tensor(ecgData).unsqueeze(0).float()

        if self.normalize:
            if self.normMethod == '0to1':
                if not torch.allclose(ecgs, torch.zeros_like(ecgs)):
                    ecgs = ecgs - torch.min(ecgs)
                    ecgs = ecgs / torch.max(ecgs)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
            elif self.normMethod == 'unitrange':
                if not torch.allclose(ecgs, torch.zeros_like(ecgs)):
                    for lead in range(ecgs.shape[0]):
                        frame = ecgs[lead]
                        frame = (frame - torch.min(frame)) / (torch.max(frame) - torch.min(frame) + 1e-8)
                        frame = frame - 0.5
                        ecgs[lead,:] = frame.unsqueeze(0)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
        
        if torch.any(torch.isnan(ecgs)):
            print(f"NANs in the data for item {item}, {ecgPath}")
        
        item = {}
        item['image'] = ecgs
        item['y'] = 1 if ejectionFraction >= 40.0 else 0
        item['key'] = 'LVEF'
        item['val'] = ejectionFraction
        item['ecgPath'] = ecgPath
        
        return item
    
    def __len__(self):
        return len(self.fileList)

