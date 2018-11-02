import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import pickle

'''
getFet function takes features(MFCC + Delta + Delta-Delta) and parameters to 
pre-process the features.
Returns the processed features according to the requirment of the model.
'''

def getFet(features,normalizer,DMfcc=False,DDMfcc=False,Energy=False):
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

	FinalFeat =  normalizer.transform(np.array(FinalFeat))

	return (FinalFeat,FinalFeat.shape[1])

#Reading trained Model
file = open('model.plk', 'r')
save = pickle.load(file)
file.close()
DMfcc,DDMfcc,Energy,NumMix = save['model']
GMM = save['GMM']
normalizer = save['normalizer']

# reading all feature MFCC,Delta Mfcc, Delta-Delta MUST have ID also
df = pd.read_hdf("./features/mfcc/timitTest.hdf")
features = np.array(df["features"].tolist())
labels = np.array(df["labels"].tolist())
Ids = np.array(df["ID"].tolist())
uniqueId = np.unique(Ids)
uniquePh = np.unique(labels)

FinalFeat,fetLen = getFet(features,normalizer,DMfcc,DDMfcc,Energy)

#Calculating Fame level predictions
prob = np.array(GMM[uniquePh[0]].score_samples(FinalFeat))
for phn in uniquePh[1:]:
	prob = np.vstack((prob,np.array(GMM[phn].score_samples(FinalFeat))))
predictions = np.argmax(prob,axis = 0)


groundTruth = []
groundPhoneData = []
predPhoneData = []
for i in range(len(uniqueId)):
	groundPhoneData.append([])
	predPhoneData.append([])

#Calculating Ground labels and data to create Files for W.E.R API call
for i in range(len(labels)):
	Id = Ids[i]
	PhoneId = list(uniquePh).index(labels[i])

	groundPhoneData[Id].append(PhoneId)
	predPhoneData[Id].append(predictions[i])
	groundTruth.append(PhoneId)

#creating Files for W.E.R API call 
GroundTruth = open('GroundTruth','w')
Pred =  open('Predictions','w')
for i in range(len(groundPhoneData)):
	gtWrite = reduce((lambda x,y: str(x)+' '+str(y)), groundPhoneData[i])
	gtWrite = str(i)+" "+gtWrite+"\n"

	pWrite = reduce((lambda x,y: str(x)+' '+str(y)), predPhoneData[i])
	pWrite = str(i)+" "+pWrite+"\n"

	GroundTruth.write(gtWrite)
	Pred.write(pWrite)
GroundTruth.close()
Pred.close()

#Calculating Frame level acuracy
score = accuracy_score(groundTruth, predictions)
print "Frame level accuracy :- ",score
