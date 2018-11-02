# ASR_2018_T01
### How to run
1) Run ```python import_timit.py --timit=./TIMIT --preprocessed=False``` computes the features.
2) Change parameters in "train.py" according to the model needed,i.e, if you want a model with MFCC+delta MFCC (and not delta-delta MFCC) and energy with 16 mixtures then make variable "DMFCC = True" ,"DDMFCC=False" , "Energy = True" and "NumMix = 16".
3) Run ``` python train.py``` .This command trains the required model.
4) Run ```python test.py``` .This command tests the trained model and returns the frame level accuracy.
5) Run ```wer GroudTruth Prediction``` . This command returns the W.E.R
## References:-

- [http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
- [http://www.pitt.edu/~naraehan/python2/pickling.html](http://www.pitt.edu/~naraehan/python2/pickling.html)
- [https://github.com/belambert/asr-evaluation](https://github.com/belambert/asr-evaluation)

## Results:-

![Results]("https://github.com/utkarsh1705/ASR_2018_T01/blob/master/p1/results/results.png")
