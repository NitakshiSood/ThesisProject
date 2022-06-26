DSAFE -
-dsafe_withoutbootstrapsampling_task2.py this file can be used for running the code.
- dsafe uses variables like RS,RF, SR, T_over,  etc for tuning.
- The dataset name is also a variable that can be set in the code.

Base learner SigDirect -
SigDirect code has only one parameter that is p-value, which doesnt need much tuning. However, config file can be used to change its value.
-The threshold for confidence values can be changed in SigDirect file, in the ripper pruning function.  (if needed)
-The fit function creates the model and  predict or predict_proba give the prediction in the form of classes and probability distributions respectively.
