import numpy as np
import torch
from torch.utils.data import Dataset
import os
import torch.nn.functional as F
# import Loader
import torchvision.transforms as transforms

class DataLoaderError(Exception):
    pass

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
        item['kclVal'] = kclVal
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