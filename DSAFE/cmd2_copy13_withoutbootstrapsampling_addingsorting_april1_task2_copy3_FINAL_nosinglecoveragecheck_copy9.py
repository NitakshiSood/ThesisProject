#  code working for boosting but testing set remains the same. 
#addedd a feature of estimator value of 0 it shoud stop the algorithm copied from version 10. does no good struggling with lists and pandas df.
#this one has has new way of findinf x and y so better than 10th version , much better thans 10 and 11. 
#This atleast runs for hepati. but some problem in classifier weight need to correct the algo.
#tryng to optimize new ensemble model as per modifications in version 12and 13 
#improving the version 9 based on boosting try 16 version. which works.
#added zero and infinity changes
#made this code for adding features to every bag to check if that helps.
#10feb 2020 -> this code only does feature selection, on say 4 divisions. there is no bagging now in this code. just 4 sub selected features perform the classification.
#this code is only successful for no missing values dataset like ionosphere. still need to do for adult.
#to make "WCB_SIGDIRECT_ensemble_onlybagging_include12features_v3copy_feb2020_10feb.py" this code work for missing data, like in adult and all 
#create features using createdatasetformissingvalues.py, which replaces all missing values to 1000. and then run this code for selective feature training.
#this code is to be used for random sampling. task 1 given on 19feb
#26feb- changing this code to check diversity only of the feature numbers and not the whole dataframe.

#this is the most latest code for selective features on 27th feb. this is prune 2 strategy, diverse code. 


# making this code on 2 mrch to automate all the experiments. for each dataset, making a list of parameter values required to be taken.
#april1 tring to make a code such that, this will see at what number of estimators/predictors is the ocmplete feature set traversed even once. and if it is less than the final number of estimators we get, them does the accuracy increase in that duration?
#now task2


import random
import numpy as np
from sklearn import svm
import math
import time
import os
import sys

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from random import randrange
from sklearn.model_selection import StratifiedKFold
#import miniproject as ml
from sklearn.utils import resample
from random import choices
import dataset_transformation as dt
from sklearn.model_selection import KFold
from sklearn.utils.multiclass import type_of_target
import csv
#import miniproject as ml
import statistics
# import Logistic as NN
# from src.NeuralNetwork import NeuralNetwork
# import src.utils as utils
#import neuralNW
import sigdirect_test
#import associative_classifier_test
#import preprocessing
import warnings
import pandas as pd
import random
from random import randint
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

count1={};
def bagging_func(y_bagging, len_pred, Ytest):
    #print("Hi")
    ypred_bagging=[]
    test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_ytestoriginal'+ str(num_run) +'.txt'
    with open(test_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dy_test = [line.strip().split(sep) for line in data]
    
    
    #y_test=dy_test
    #Ytest=pd.DataFrame(dy_test)
    Ytest=pd.DataFrame(Ytest)
    for j in range(0,len_pred):
        temp=[]
        for i in range(len(y_bagging)):
            temp.append(y_bagging[i][j])
        ypred_bagging.append(temp)
    
    #print("!!!!!!!!!! ypred_bagging !!!!!!!!!!!!") 
    #print(ypred_bagging)
    
    ypred_final=[]
    for i in range(len(ypred_bagging)):
        mode_val=max(set(ypred_bagging[i]), key=ypred_bagging[i].count)
        ypred_final.append(mode_val)        
        
    #print("ypred_final")
    #print(ypred_final)
    #print("len(ypred_final): ",type(ypred_final))
    #print("Ytest")
    #print(Ytest)    
    #print("len(Ytest): ",type(Ytest))
    count=0
    for i in range(len(ypred_final)):
        if(ypred_final[i]==Ytest[0][i]):
            count=count+1
    #print("count: ",count)
    print("final bagging accuracy: ", (count/len(ypred_final)))
    #print(stopp)
    return (count/len(ypred_final))

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    random.shuffle(dataset)
    temp=[]
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        #index = randint(0,len(dataset)-1)
        temp.append(index)
        sample.append(dataset[index])
    #print("#######temp:")
    #print(temp)
    return sample

# Create a random subsample from the dataset with replacement but no repetition
def subsample_features(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    '''
    print("----------------------FOCUS HERE------------------")
    print("ratio: ",ratio)
    print("len(dataset): ",len(dataset))
    print("n_sample: ",n_sample)
    print("-----------END--------------------")
    '''
    random.shuffle(dataset)
    #print("----------------------------DOES RANDOM SHUFFLING HELP?: ",dataset)
    temp=[]
    tempset=set() # to ensure no repetition
    while len(sample) < n_sample:
        index = randrange(len(dataset))        
        temp.append(index)
        if dataset[index] in sample:
            continue
        else:
            sample.append(dataset[index])
    #print("#######temp:")
    #print(temp)
    return sample

def subsample_features_checkeachstep(dataset, ratio,T_over,diverseSets):
    sample = list()
    #diverseSets={}
    change=0
    n_sample = round(len(dataset) * ratio)
    '''
    print("----------------------FOCUS HERE------------------")
    print("ratio: ",ratio)
    print("len(dataset): ",len(dataset))
    print("n_sample: ",n_sample)
    print("-----------END--------------------")
    '''
    random.shuffle(dataset)
    #print("----------------------------DOES RANDOM SHUFFLING HELP?: ",dataset)
    temp=[]
    tempset=set() # to ensure no repetition
    somecount=0
    while len(sample) < n_sample:
        index = randrange(len(dataset))        
        temp.append(index)
        if dataset[index] in sample:
            continue
        else:
            #here you should check if adding it would change the overlap rate check with threeshold -> t_over
            #isfeatureframeDiverse(X_train_sub,diverseSets,T_over): call toh ye hoga pr diverset ka kya kregi?
            if len(sample)>=n_sample*T_over: #n_sample/2 #set this value as RF or RS this sohuld be ideally equal to t_over rate......
                index2 = randrange(len(dataset))
                if dataset[index2] in sample or index2==index:
                    
                    continue
                else:
                    #print("n_sample: ",n_sample)
                    #print("####sample: ",sample)
                    #print("NOW SHOULD WE ADD: ",dataset[index])
                    sample_try=sample.copy()
                    sample_try.append(dataset[index])
                    sample_try.append(dataset[index2])
                    i=len(sample_try)
                    while i<n_sample:
                        sample_try.append(999)
                        i+=1
                    #print("@@@@@@@@@@sample_try: ",sample_try)
                    
                    #,X_train_subsample,diverseSets
                    sample_try_df=pd.DataFrame(sample_try)
                    if isfeatureframeDiverse(sample_try_df,diverseSets,T_over):
                        #print("HERE")
                        #print(stoppppp)
                        sample.append(dataset[index])
                        sample.append(dataset[index2])
                        #if len(sample)==n_sample/2:
                        #    diverseSets[count]=sample
                        #    count+=1
                        #print("sample???????????: ",sample)
                    else:
                        #change=change+1
                        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO") #is this now coming only during the last entry.
                        return []
                        #print(stopp)
                        #if change==100 : #sr stagnation rate
                        #    return []
            else:
                sample.append(dataset[index])
    #print("#######temp:")
    #print(temp)
    #print("?????????????????????????????????diverseSets: ",diverseSets)
    #print("sample???????????now: ",sample)
    #print(stoppppp)
    return sample




def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
     



def unique(list1): 
    x = np.array(list1) 
    print(np.unique(x)) 

 

def f(x):
    if x.last_valid_index() is None:
        return np.nan
    else:
        return x[x.last_valid_index()]

def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]

from random import shuffle

def split(data, n):   
    size=int(len(data)/n);
    for i in range(0, n*size, size):
        yield data[i:i+size]
        

def isfeatureframeDiverse(X_train_sub,diverseSets,T_over):
    #print("this is is feature frame diverse function ")
    
    var=True  
        
    #print("len(diverseSets): ",len(diverseSets))
    if len(diverseSets)==0 :
        #print("YAHAN SE GAYA KYA???")
        return True
    else:
        #print("IN ELSE LOOP:")
        flag=0
        for var in diverseSets.keys(): 
            
            #print("diverseSets: ",diverseSets)
            #print("COMPARING WITHHHHHHH X_train_sub : ",X_train_sub )
            #print("type of X_train_sub: ", type(diverseSets[var]))
            #print()
                      
            #problem is how to find intersection???????? because order is not same.
            #print(X_train_sub.merge(diverseSets[var]))
            #print("X_train_sub: ",X_train_sub)
            
            X_train_subcopy=X_train_sub.sort_values(by=0,axis=0, ascending=True)
            #print("DID IT GET SORTED????? X_train_sub: ",X_train_subcopy)
            #cols = X_train_subcopy.columns.tolist()
            #print("before cols: ",cols,cols[0])
            #print(cols[1:],cols[0])
            
            #newcollist=cols[1:]
            #newcollist.append(cols[0])
            #print("after cols: ",newcollist)
            #X_train_sub = X_train_subcopy.reindex(columns= newcollist) # done this basically first col was not geting sorted.
            #print(X_train_sub)
            #X_train_sub.columns = list(range(0, len(newcollist)))
            #print('now? : ',X_train_sub)
            #print("comparing with diverseSets: ",diverseSets[var])
            #print(stopphere)
            #print(">1???????????????????",X_train_sub.shape[0],X_train_sub.shape[1])
            #print(">2???????????????????",diverseSets[var].shape[0],diverseSets[var].shape[1])
            #print("COUNT OF INTERSECTION: ",len(X_train_sub.merge(diverseSets[var])),X_train_sub.shape[0],X_train_sub.shape[1])
            #print(stophere) 
            
            #print("check the length of intersection: ",len(X_train_sub.merge(diverseSets[var])))
            #print("diverseSets[var]: ",diverseSets[var])
            #print(X_train_sub)
            overlap_rate=len(X_train_sub.merge(diverseSets[var]))/(X_train_sub.shape[0]*X_train_sub.shape[1])
            #print("overlap_rate: ",overlap_rate)
             
            #if len(X_train_sub.merge(diverseSets[var]))>0:
            #    print(possiblehaikya)
            #print("overlap_rate: T_over: ",overlap_rate,T_over)
            if overlap_rate>T_over:
                #print("yahan gaya tha???")
                #savevaloverlap_rate=overlap_rate
                #savevalT_over=T_over
                flag=1
                break
        if flag==1:
            #print("overlap_rat???????e: T_over: ",savevaloverlap_rate,savevalT_over)
            #print("GOES IN HERE")
            #print(isdiverse)
            return False
        
        else:
            #print(isdiverse2)
            return True    
    
        
def GenerateDiverseSets(X_train,Xtotal_subsample,X_test,Xval_subsample,RS, RF, T_over, sr):
    #print("this func is for  GenerateDiverseSets ")
    change=0
    diverseSets={}
    feature_split_original=list(range(0, X_train.shape[1]))
    print("***************feature_split_original****************************: ",feature_split_original)
    count=1
    coverage_set_count=1
    class_label_col=Xtotal_subsample.shape[1]-1
    #print("num_features: ",class_label_col)
    Xtotal_subsample_list=Xtotal_subsample.values.tolist() # xtrain total it is
    Xtrain_subsample_list=feature_split_original#X_train.values.tolist()
    #print("Xtrain_subsample_list: ",Xtrain_subsample_list)
    df1=pd.DataFrame(X_train)
    #print()
    df1_test=pd.DataFrame(X_test)
    df1_val=pd.DataFrame(Xval_subsample)
    #print()
    coverage_set=set()
    flag=0
    while change<sr:# or count!=4:
            if count>=100:
                return count,coverage_set_count
            #if count==50:
            #    print(diverseSets)
            #    print(stopp)
            
            Xtotal=Xtotal_subsample
            
            #feature subselection        
            random.shuffle(Xtrain_subsample_list)
            #sample=subsample_features(Xtrain_subsample_list, RF) 
            #print("sample: ",sample)
            
            #print("NOWWWsample?????????? ",sample)
            #print("-________________~~~~~~~~~~~~~~~~~~~~~~~~~~~VALUE OF CHANGE IS: ",change,sr,count)
            sample=subsample_features_checkeachstep(Xtrain_subsample_list, RF,T_over,diverseSets) 
            if sample==[]:
                change=change+1
                continue
                   
            X_train_subsample=pd.DataFrame(sample)
            #print("X_train_subsample: ",X_train_subsample)        
            sample.sort()
            if isfeatureframeDiverse(X_train_subsample,diverseSets,T_over):
                
                
                #coverage_set=coverage_set|set(sample)
                
                #if sorted(feature_split_original)==sorted(list(coverage_set)) and flag==0:
                    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~*****************count: ",count)
                 #   coverage_set_count=count
                 #   flag=1
                #print(stop)
                ########################
                diverseSets[count]=X_train_subsample
                
                print("*****************************************count: ",count)
                
                #count=count+1
                #print(baggingsample)
                change=0 
                
                ##################################################
                
                sample.append(int(class_label_col))
                #print("sample?????????? ",sample)           
                    
                #creating file
                X_train_sub=pd.DataFrame()
                X_test_sub=pd.DataFrame() 
                X_val_sub=pd.DataFrame()
                #print("df1_test: ",df1_test)
                #print("feature_split[tempp]: ",feature_split[tempp])
                #print(stopp)   
                for temp3 in sample:# (feature_split[tempp]):                                               
                    X_train_sub[temp3]=Xtotal.iloc[:,temp3]
                    X_test_sub[temp3]=df1_test.iloc[:,temp3]
                    X_val_sub[temp3]=df1_val.iloc[:,temp3]
                    
                X_train_sub=X_train_sub.replace('', np.NaN).dropna(how='all') 
                X_train_sub=X_train_sub.where(pd.notnull(X_train_sub), None)
                X_train_sub=X_train_sub.values.tolist()
                for k in range(len(X_train_sub)):
                    
                    X_train_sub[k]=[x for x in X_train_sub[k] if x is not None]            
                X_train_sub=pd.DataFrame(X_train_sub)
                mask = X_train_sub.applymap(lambda x: x is None)
                cols = X_train_sub.columns[(mask).any()]
                for col in X_train_sub[cols]:
                    X_train_sub.loc[mask[col], col] = ''         
                
               
                
                #print(stopp)
                # same for test set
                
                X_test_sub=X_test_sub.replace('', np.NaN).dropna(how='all') 
                X_test_sub=X_test_sub.where(pd.notnull(X_test_sub), None)
                X_test_sub=X_test_sub.values.tolist()
                for k in range(len(X_test_sub)):
                    
                    X_test_sub[k]=[x for x in X_test_sub[k] if x is not None]            
                X_test_sub=pd.DataFrame(X_test_sub)
                mask = X_test_sub.applymap(lambda x: x is None)
                cols = X_test_sub.columns[(mask).any()]
                for col in X_test_sub[cols]:
                    X_test_sub.loc[mask[col], col] = ''         
                
               
                #print("X_test_sub: ",X_test_sub)            
                
                train_name  =   'datasets_bag_rc/' + dataset_name + '_train'+ str(count) +'.txt'
                test_name  =   'datasets_bag_rc/' + dataset_name + '_test'+ str(count) +'.txt'
                
                X_train_sub.to_csv(train_name,sep=' ',index=False,header=False)   
                X_test_sub.to_csv(test_name,sep=' ',index=False,header=False) 
                
                
                 #same for val set
                
                X_val_sub=X_val_sub.replace('', np.NaN).dropna(how='all') 
                X_val_sub=X_val_sub.where(pd.notnull(X_val_sub), None)
                X_val_sub=X_val_sub.values.tolist()
                for k in range(len(X_val_sub)):
                    
                    X_val_sub[k]=[x for x in X_val_sub[k] if x is not None]            
                X_val_sub=pd.DataFrame(X_val_sub)
                mask = X_val_sub.applymap(lambda x: x is None)
                cols = X_val_sub.columns[(mask).any()]
                for col in X_val_sub[cols]:
                    X_val_sub.loc[mask[col], col] = ''         
                
               
                #print("X_val_sub: ",X_val_sub)            
                
                #train_name  =   'datasets_bag_rc/' + dataset_name + '_train'+ str(count) +'.txt'
                val_name  =   'datasets_bag_rc/' + dataset_name + '_val'+ str(count) +'.txt'
                
                #X_train_sub.to_csv(train_name,sep=' ',index=False,header=False)   
                X_val_sub.to_csv(val_name,sep=' ',index=False,header=False)  
                
                if count==1:
                    print("SIZE OF X_train_sub: ",X_train_sub.shape)
                    print("SIZE OF X_val_sub: ",X_val_sub.shape)
                    print("SIZE OF X_test_sub: ",X_test_sub.shape)
                    #print("X_train_sub: ")
                    #print(X_train_sub)
                    #print("X_val_sub: ")
                    #print(X_val_sub)
                    #print("X_test_sub: ")
                    #print(X_test_sub)                
                    #print(stopby) 
                #print("X_test_sub")
                #print(X_test_sub)
                #print(stop)
                #######################################################
                count=count+1
                
            else:
                
                #print("not diverse and change count is: ",change)
                change=change+1
                #print(stoppheeere)        
        
        
        
        #print(baggingsample)
    print("value of count is~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~: ",count-1)
    print("-----------------------------------TO CHECK COVERAGE : diverseSets: ")
    
    #print("***************feature_split_original****************************: ",feature_split_original)
    #print("all the features were covered at: ",coverage_set_count)
    #print("coverage_set: ",coverage_set)
    #print(nowstop)
    #print(stopphere)
    #print(stopp)
        
    
    for key in diverseSets:
        print()
        print(diverseSets[key].T)
        
        print("  DONE    ")    
    #print(stophere)
    return count,coverage_set_count
    
#in this code add two things- one for random mixing of features, then data transform ko call karo. and then, check.  
# ye sirf order me divide krta hai features ko.
if __name__ == "__main__":
    
    # dataSets=['iris', 'monks','transfusion','yeast','tic-tac-toe','diagnosis','facebook','horse-colic','anneal']
    #dataSets = ['adult']#,'iris', 'pima-indians-diabetes']
    test_count = 1
    
    #for fileName in dataSets:
    #    print("Running for dataset: ", fileName)
    #edited_WATT_Data7_March14_2018_preprocessed_NumNomConvert_3divnormcopy
    #WCB_SIGDIRECT_ensemble_onlybagging_include12features_v3
    fileName='data_covid_smoted_DISCRETIZED_completedata'#'adult_arff_nomissingval'#'ionosphere'#cylband_nomissingdata5'#'penDigits'#'cylband_nomissingdata5'#'ionosphere'
    #soybean_nomissingfile'#'ionosphere'#soybean_nomissingfile'#adultnew'#ionosphere'#adult'#'edited_WATT_Data7_March14_2018_preprocessed_NumNomConvert_3divnorm' #WCB_V5_norm _copy
    dataset_name=fileName
    
    
    num_runs=1
    num_estimator_avg=[]
    eachrunoutput=[]
    #assert len(sys.argv)>1
    dataset_name = sys.argv[1]
    print("Running for dataset: ", dataset_name)
    #RS=float(sys.argv[2])
    #RF=float(sys.argv[3])
    #T_over=float(sys.argv[4])
    #sr=int(sys.argv[5])
    #RS=0.7, RF=0.5, T_over=0.65, sr=100
    RS=1.0
    RF=0.6
    sr=100
    #print(dataset_name,RS,RF,T_over,sr)
    #print(stop)
    
    #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    T_over_list=[0.6]#[0.1,0.2,0.3,0.4]#,0.2,0.3,0.4,0.5,0.6,0.7,0.8] #[0.6]#,0.2,0.3,0.4,0.5,0.6,0.7,0.8]#[0.8]
    res_dict={}
    for T_over in T_over_list: 
        num_estimator_avg=[]
        eachrunoutput=[]
        num_estimator_avg_coverage_set_count=[]
        eachrunoutput_coverage_set_count=[]
        tt2 = time.time()
        num_runs=10
        print(dataset_name,RS,RF,T_over,sr)
        for runnumber in range(num_runs):
            num_run=1 # just keep this variable as is so that results dont get wrong. as per the below code.
            print("How many time: run number: ",runnumber)
            input_name  = 'datasets_new_originalfiles/' + dataset_name + '.txt' #train file
            
            nameFile="datasets_new_originalfiles" +"\\"+fileName+".names"
            
            #method 1 for readin using pandas dataframe
            sep = ' '
            
            with open(input_name, 'r') as f:
                data = f.read().strip().split('\n')
            dataset = [line.strip().split(sep) for line in data]        
            
           
            df=pd.DataFrame(dataset)
            #print("df:",df )
            dforiginal=df
            #masking all the nones if any
            mask = df.applymap(lambda x: x is None)
            cols = df.columns[(mask).any()]
            for col in df[cols]:
                df.loc[mask[col], col] = ''
            dforiginal=df    
             
            dfnew=df
            #X = df.iloc[:,:-1].values
            #this line replaces all empty spaces with nan. this is done to get the last col values.
            dfnew = dfnew.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('', np.nan)
            
            #now u can get the last col values that is the labels.
            Y=dfnew.groupby(['label'] * dfnew.shape[1], 1).agg('last')
            
            dfnew=dfnew.where(pd.notnull(dfnew), None)
            #dfnew = dfnew.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace(np.nan,None)
            dfnew=dfnew.values.tolist()
            
            print()
            for k in range(len(dfnew)):
                dfnew[k]=[x for x in dfnew[k] if x is not None]        #removing none from list
            
           
            
            print()
            for i in dfnew:
                i.pop()
            X=dfnew    
            
           
            #remove none and then then pop out the last element
            #masking all the nones if any
            
            
            #Y = df.iloc[:,-1].values # Here first : means fetch all rows :-1 means except last column
            #X = df.iloc[:,:-1].values # : is fetch all rows 3 means 3rd column  
            
            #print("len(X)",len(X))
            #print("len(Y)",len(Y))
            
            #X=np.array([np.array(xi) for xi in X])
            #X=np.asarray(X) 
            #print("Now x: ",X)
            
            X=pd.DataFrame(X)
            
            mask = X.applymap(lambda x: x is None)
            cols = X.columns[(mask).any()]
            for col in X[cols]:
                X.loc[mask[col], col] = '' 
           
            
            print("X: ",X)
            print("Y: ",Y)
            print("Len of X: ",X.shape)
            print("Len of Y: ",Y.shape)
            print("30% OF TOTAL width: ",(X.shape[1]*30)/100)
            print("50% OF TOTAL width: ",(X.shape[1]*50)/100)
            print("70% OF TOTAL width: ",(X.shape[1]*70)/100)
            
            
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle='true')
            
            #now split the prune set
                 
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, shuffle='true')
            #X_val=X_train
            #y_val=y_train
            #print(X_train)
            #print(y_train)
            '''
            print("###########################len(X_train): ",len(X_train))
            
            print("len(X_test): ",len(X_test))
            print("len(y_train): ",len(y_train))
            print("len(y_test): ",len(y_test))
            print("len(X_val): ",len(X_val))
            print("len(y_val): ",len(y_val))
            '''
            X_test_original=X_test
            #df3_test=pd.DataFrame(list(X_test_original))              
            df_Xtest=X_test_original              
            test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_testoriginal'+ str(num_run) +'.txt'
            df_Xtest.to_csv(test_name,sep=' ',index=False,header=False)  
            
            df3_test=y_test 
            test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_ytestoriginal'+ str(num_run) +'.txt'
            df3_test.to_csv(test_name,sep=' ',index=False,header=False)          
                    
            df_Xtrain=X_train                     
            test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_trainoriginal'+ str(num_run) +'.txt'
            df_Xtrain.to_csv(test_name,sep=' ',index=False,header=False)
            
            df3_test=y_train                     
            test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_ytrainoriginal'+ str(num_run) +'.txt'
            df3_test.to_csv(test_name,sep=' ',index=False,header=False)        
            
            #df1=pd.DataFrame(list(X_train))
            #df2=pd.DataFrame(list(y_train)) 
            #print(X_train)
            #print(y_train)
            df1=pd.DataFrame(X_train)
            df2=pd.DataFrame(y_train)   
            
            #for val set
            df1_val=pd.DataFrame(X_val)
            df2_val=pd.DataFrame(y_val)
            Xval_subsample = pd.concat([df1_val,df2_val], axis=1)
            Xval_subsample=Xval_subsample.reset_index(drop=True)
            Xval_subsample.columns = list(range(0, X_train.shape[1]+1))        
            #print("Xval_subsample: ",Xval_subsample)       
            #print("NOW??????????????????df1: ",df1)
            
            
            df4y_test=pd.DataFrame(X_test) 
            y_test=pd.DataFrame(y_test)
            Xtest_subsample = pd.concat([df4y_test,y_test], axis=1)
            Xtest_subsample=Xtest_subsample.reset_index(drop=True)
            Xtest_subsample.columns = list(range(0, X_train.shape[1]+1))
            #print("Xtest_subsample: ",Xtest_subsample)
            #print("df1: ",df1)
            #print("df2: ",df2)
            
            Xtotal_subsample = pd.concat([df1,df2], axis=1)
            #print(breakhere) 
            
            #print("Xtotal_subsample: ",len(Xtotal_subsample),len(df1))
            Xtotal_subsample=Xtotal_subsample.reset_index(drop=True)
            Xtotal_subsample.columns = list(range(0, X_train.shape[1]+1))
            #print("Xtotal_subsample: ",Xtotal_subsample)
            #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            #print(stoppppppp)
            
            df1=pd.DataFrame(X_train)
            df2=pd.DataFrame(y_train)        
            
            
            #to create bags of features
            bag=[]
            
            #try this
            print("*************FINAL********************")
            feature_split=list(range(0, X_train.shape[1]))
            feature_split_original=list(range(0, X_train.shape[1]))
            #print(feature_split)
            shuffle(feature_split)
            #print(feature_split)
            feature_split=list(split(feature_split, 4)) #set this split variable value
            #print("rand_list??????: ",feature_split)
            #print(stopp)
            X_train_sub=[]
            X_test_sub=[]  
            #print("df1: ")
            #print(df1)
            #print("df4y_test: ")
            #print(df4y_test)    
            #print("feature_split: ",feature_split)
            #print(len(feature_split))
            
            #num_estimator=GenerateDiverseSets(X_train,Xtotal_subsample,Xtest_subsample,Xval_subsample,RS=0.7, RF=0.5, T_over=0.65, sr=100)
            
            num_estimator,coverage_set_count=GenerateDiverseSets(X_train,Xtotal_subsample,Xtest_subsample,Xval_subsample,RS, RF, T_over, sr)
            print(" ################################ ")
            #print(stophere)
            #num_estimator=100
            y_bagging=[]
            y_bagging_coverage_set_count=[]
            for var_i in range(1,num_estimator):#+1
                print("Number is: ",var_i)
                predictions,y_test_innewform=sigdirect_test.main(dataset_name,var_i)
                #print("predictions: ",predictions)
                #print("type(predictions: )", type(predictions))
                
                y_bagging.append(predictions)
                #if var_i<=coverage_set_count-1:
                #    y_bagging_coverage_set_count.append(predictions)
                
            
            #print("y_bagging: ",y_bagging)
            print("###########################Calling bagging for all estimators: ")
            out=bagging_func(y_bagging, len(X_test), y_test_innewform)# y_test)
            print("###########################Calling bagging for coverage_set_count: ")
            #out_coverage_set_count=bagging_func(y_bagging_coverage_set_count, len(X_test), y_test_innewform)# y_test) 
            #print("out???????????????????????????: ",out,out_coverage_set_count)
            #print(stopp)
            eachrunoutput.append(out)
            num_estimator_avg.append(num_estimator)
            #print(stop)
            #eachrunoutput_coverage_set_count.append(out_coverage_set_count)
            #num_estimator_avg_coverage_set_count.append(coverage_set_count)
            #print(stopp)
                
            
        
        #print(stopp)
        #print("variables were RS, RF, T_over, sr: ",RS, RF, T_over, sr)
        res_dict[T_over]=[sum(eachrunoutput)/len(eachrunoutput),sum(num_estimator_avg)/len(num_estimator_avg)]
        print("------------------------------------------------RESULT SHEET -----------------------------------------------------")
        print("Dataset and variables were RS, RF, T_over, sr",dataset_name,RS,RF,T_over,sr)
        print("list of num_estimator_avg: ",num_estimator_avg)
        print("num_estimator_avg: ",sum(num_estimator_avg)/len(num_estimator_avg))
        print("of all the runs: ",eachrunoutput)
        print("Final avg accuracy: ", sum(eachrunoutput)/len(eachrunoutput))  
        print('FINAL TOTAL TIME:', time.time()-tt2)
        print("------------------------------------------------RESULT SHEET ENDS-----------------------------------------------------")
        #print("------------------------------------------------RESULT SHEET FOR coverage count parameter-----------------------------------------------------")
        #print("Dataset and variables were RS, RF, T_over, sr",dataset_name,RS,RF,T_over,sr)
        #print("list of num_estimator_avg: ",num_estimator_avg_coverage_set_count)
        #print("num_estimator_avg: ",sum(num_estimator_avg_coverage_set_count)/len(num_estimator_avg_coverage_set_count))
        #print("of all the runs: ",eachrunoutput_coverage_set_count)
        #print("Final avg accuracy: ", sum(eachrunoutput_coverage_set_count)/len(eachrunoutput_coverage_set_count))  
        #print('FINAL TOTAL TIME:', time.time()-tt2)
        #print("------------------------------------------------RESULT SHEET ENDS-----------------------------------------------------")        
        
    print("FINAL res_dict: ",res_dict)   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            
            

        


