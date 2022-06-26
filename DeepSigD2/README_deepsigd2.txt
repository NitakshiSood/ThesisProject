DeepSigD2 -
- DeepSigD2.py is the mail file can be used for running the code.
- windowsize_list is the list where different window sizes can be varied for tuning.
- The dataset name is also a variable that can be set in the code.
- Probability score can be obtained from predict_proba function in sigdirect.py

Base learner SigDirect -
SigDirect code has only one parameter that is p-value, which doesnt need much tuning. However, config file can be used to change its value.
-The threshold for confidence values can be changed in SigDirect file, in the ripper pruning function.  (if needed)
-The fit function creates the model and  predict or predict_proba give the prediction in the form of classes and probability distributions respectively.
