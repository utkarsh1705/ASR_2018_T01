import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.mixture import GaussianMixture
import pickle

'''
getFet function takes features(MFCC + Delta + Delta-Delta) and parameters to 
pre-process the features.
Returns the processed features according to the requirment of the model.
'''

def getFet(features,DMfcc=False,DDMfcc=False,Energy=False):
	featuresMfcc,featuresDMfcc,featuresDDMfcc = np.hsplit(features,3)
	if(Energy==False):
		featuresMfcc = np.delete(featuresMfcc,0,1)
		featuresDMfcc = np.delete(featuresDMfcc,0,1)
		featuresDDMfcc = np.delete(featuresDDMfcc,0,1)

	FinalFeat = featuresMfcc
	if(DMfcc==True):
		FinalFeat = np.concatenate((FinalFeat,featuresDMfcc),axis = 1)
	if(DDMfcc==True):
		FinalFeat = np.concatenate((FinalFeat,featuresDDMfcc),axis = 1)

	normalizer = Normalizer()
	FinalFeat = normalizer.fit_transform(np.array(FinalFeat))
	
	phon = {}
	for frame in range(len(FinalFeat)):
		if  labels[frame] not in phon:
			phon[labels[frame]] = [FinalFeat[frame]]
		else:
			phon[labels[frame]].append(FinalFeat[frame])
	
	return (phon,normalizer,FinalFeat.shape[1])
#parameters to pre-process the features
DMfcc = True
DDMfcc = True
Energy = True

#number of mixtures in GMM
NumMix = 4

df = pd.read_hdf("./features/mfcc/timitTrain.hdf")# reading all feature MFCC,Delta Mfcc, Delta-Delta

features = np.array(df["features"].tolist())
labels = np.array(df["labels"].tolist())
uniquePh = np.unique(labels)

phon,normalizer,fetLen = getFet(features,DMfcc,DDMfcc,Energy)

#Training GMM for each Phoneme
GMM = {}
for phn in phon.keys():
    GMM[phn] = GaussianMixture(n_components=NumMix, covariance_type='diag')
    GMM[phn].fit(phon[phn])

# model Info
model = [DMfcc,DDMfcc,Energy,NumMix] 
save = {}
save['model'] = model
save['normalizer'] = normalizer
save['GMM'] = GMM
#Saving the model
file = open('model.plk', 'w')
pickle.dump(save, file)
file.close()