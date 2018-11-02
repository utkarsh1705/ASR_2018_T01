# ASR_2018_T01
### How to run
1) Run ```python import_timit.py --timit=./TIMIT --preprocessed=False```
2) Change parameters in "train.py" according to the model needed,i.e, if you want a model with MFCC+delta MFCC (and not delta-delta MFCC) and energy with 16 mixtures then make variable "DMFCC = true" ,"DDMFCC=False" , "Energy = true" and "NumMix = 16".
3) Run ``` python train.p```
4) Run ```python test.py```
5) Run ```wer GroudTruth Prediction```
