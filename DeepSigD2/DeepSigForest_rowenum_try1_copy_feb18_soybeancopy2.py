#try to add more sliding windows
#lets randomly sub select the features with replacement using subsample function in this script
#trying to add more sig classifierers in cascade stage. as it is bagging.
#Sigdirect_phase3_v2.4 without extra print statements
#this code adds the testing part
#this code works for val and testing. just improving the last stage of testing here.taking max of all the forests in last stage. 
#this code worked finally on 7th nov. seems to be complete
import random
import numpy as np
from sklearn import svm
#from imblearn.metrics import sensitivity_specificity_support
import gc
import sys    
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from random import randrange
from sklearn.model_selection import StratifiedKFold
#import miniproject as ml
import dataset_transformation as dataset_transformation
import dataset_transformation_mgstage1 as dataset_transformation_mgstage1
from sklearn.utils import resample
from random import choices
from sklearn.model_selection import KFold
from sklearn.utils.multiclass import type_of_target
import csv
#import miniproject as ml
import statistics
# import Logistic as NN
# from src.NeuralNetwork import NeuralNetwork
# import src.utils as utils
#import neuralNW
#import associative_classifier_test
import sigdirect_test
#import preprocessing
import warnings
import pandas as pd
import math
import random
from random import randint
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
sum_dictvalues=0;
count1={};
def cascade_op_valstep(new_val_train,new_val,n,df_y_train,df_y_val,test_name):
    
    # this function only creates file for associaitive classifier nothing else
    
    #finalchunksublist,finalchunksublist_new_val=chunks(new_val_train,new_val, n)
    
    finalchunksublist,finalchunksublist_new_val=bagged_chunks(new_val_train,new_val, n)
    #print("finalchunksublist: ",finalchunksublist[1])
    #print("finalchunksublist_new_val: ",finalchunksublist_new_val[1])
    
    #cascades_classifiers()
    count=0
    for chunknum in range(len(finalchunksublist)):
        df1_train=finalchunksublist[chunknum]
        df2=df_y_train
        
        df3_train=finalchunksublist_new_val[chunknum]
        df4=df_y_val
        df1_train,df2,df3_train,df4=dataset_transformation.main(df1_train,df2,df3_train,df4)
        #print(stopcheckcheck)
        #for train set
        #print("df1_train>?: ",df1_train)
        #print("df2:;;;; ",df2)
        Xtotal_subsample = pd.concat([df1_train,df2], axis=1)
        #print("?Xtotal_subsample1: ",Xtotal_subsample)
        #print(weight22)
        Xtest_total=Xtotal_subsample.values.tolist()         
        #print("Xtest_total: ")
        #test_name  = 'subset_datasets/' + dataset_name + '_train'+ str(var) +'.txt'
        test_name  = 'subset_datasets/' + dataset_name + '_train'+ str(count) +'.txt'
        
        for var in range(len(Xtest_total)):
            
            Xtest_total[var] = [x for x in Xtest_total[var] if str(x) != 'nan' and str(x) != '']         
        
        import csv
        with open(test_name, "w",newline='\n') as f:
            writer = csv.writer(f,delimiter=' ')
            writer.writerows(Xtest_total) 
        #forval set    
        Xtotal_subsample = pd.concat([df3_train,df4], axis=1) 
        #print("df3_train??: ",df3_train)
        #print("df4??: ",df4)
        #print("Xtotal_subsample: ",Xtotal_subsample)
        #Xtotal_subsample=dataset_transformation.main(Xtotal_subsample)
        #print("?Xtotal_subsample2: ",Xtotal_subsample)
        Xtest_total=Xtotal_subsample.values.tolist()         
        #print("Xtest_total: ")
        #test_name  = 'subset_datasets/' + dataset_name + '_train'+ str(var) +'.txt'
        test_name  = 'subset_datasets/' + dataset_name + '_val'+ str(count) +'.txt'
       
        for var in range(len(Xtest_total)):
            
            Xtest_total[var] = [x for x in Xtest_total[var] if str(x) != 'nan' and str(x) != '']         
        
        import csv
        with open(test_name, "w",newline='\n') as f:
            writer = csv.writer(f,delimiter=' ')
            writer.writerows(Xtest_total)
        count+=1
    #print(stopnowcheckcheck)   
    
    #print(weight22)        
    #print(weight22)
   
def cascadestep2_valstep(new_val_train,new_val,n,df_X_val,df_X_train,df_y_train,df_y_val,test_name,dataset_name,df2_slidingwindow,df3_slidingwindow,layer,num_layers,flagstage): 
    print("GOES HERE?")
    #here you need to run this code for 4 sigforests 
    num_sigforests=4 #number of sigs you want in cascade step
    new_val_train_original=new_val_train
    new_val_original=new_val
    df_y_train_original=df_y_train
    df_y_val_original=df_y_val
    flag=0
    lastlayer_pred_list=[]
    for num in range(num_sigforests):
        cascade_op_valstep(new_val_train_original,new_val_original,n,df_y_train_original,df_y_val_original,test_name)
        #cascade_op(new_val_train,new_val,n,df_y_train,df_y_val,test_name)
        #calling a subsig
        for subsig in range(n):
            
            input_name  = 'subset_datasets/' + dataset_name + '_val'+ str(subsig)+'.txt'
            test_name   = 'subset_datasets/' + dataset_name + '_train' + str(subsig) + '.txt'
            
            #print()
            predictions, acc,g_rules_count,np_rules_count,hrs_acc,scores,scores_testset=sigdirect_test.main(dataset_name,subsig,input_name,test_name)
            if layer==(num_layers-1) and flagstage==0:
                print("Gores in here?")
                lastlayer_pred_list.append(predictions)
            #print(stophere)
            #print("?????scores: ",type(scores))
            
            if subsig==0 and flag==0:
                df2=scores
                df3=scores_testset #SOME PROBLEM THERE, SOME SCORES ARE 0 FOR ALL. CHECK THIS OUT.
                flag=1
                #print("df2: ",df2)
                #print("df3: ",df3)            
                #print(weigh)
            else:
                df2 = pd.concat([df2, scores], axis=0, sort=False)
                df2=df2.groupby(level=0).mean()
                df3 = pd.concat([df3, scores_testset], axis=0, sort=False)
                df3=df3.groupby(level=0).mean()            
            print("?????@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2")
    #if layer
    #print("lastlayer_pred_list: ",lastlayer_pred_list)
    #print()
    #print("df2: ",df2)
    #print("df3: ",df3)
    #df2=df2.to_frame() 
    #print("????: ",type(df2))
    
    #print(type(df2),type(df_X_val))
    #print(df_X_val)
    #print(df2)
    df_y_val=df_y_val.reset_index(drop=True)
    df_X_val=df_X_val.reset_index(drop=True)
    df_y_train=df_y_train.reset_index(drop=True)
    df_X_train=df_X_train.reset_index(drop=True)        
    df2=df2.reset_index(drop=True)
    df2=df2.T
    df3=df3.reset_index(drop=True)
    df3=df3.T        
       
    new_val=pd.concat([df_X_val,df2_slidingwindow, df2,df_y_val], axis=1, sort=False) # ihave to do the avg here
    
    #print("df_X_val:df2 ",df_X_val.shape, df2.shape,new_val.shape)
    #print("new val set: ",new_val)
    #print("sig_num: ",sig_num)
    
    new_val_train= pd.concat([df_X_train,df3_slidingwindow, df3,df_y_train], axis=1, sort=False)    
    
    return new_val_train,new_val,lastlayer_pred_list
def merge(list1, list2): 
      
    merged_list = list(zip(list1, list2))
    return merged_list

def unique(list1): 
    x = np.array(list1) 
    print(np.unique(x)) 

def f(x):
    if x.last_valid_index() is None:
        return np.nan
    else:
        return x[x.last_valid_index()]
    
def Merge(dict1, dict2): 
    return(dict2.update(dict1)) 

from itertools import islice

def rolling_window(a, window):
    '''
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    '''
    
    #for
    Xtrain=a
    Xtrain_sub=[]
    for var in range(Xtrain.shape[1]-window): #Xtrain.shape[1]-window+1
        #print(Xtrain[:,var:var+window-1])
        #Xtrain_sub.append(Xtrain[:,var:var+window-1])
        
        #print(Xtrain[:,var:var+window])
        Xtrain_sub.append(Xtrain[:,var:var+window])
        #print(stop)
    Xtrain_sub.append(Xtrain[:,var:])
    #print("Xtrain in rolling window: ",Xtrain_sub)
    
    #print(weighh)
    return Xtrain_sub

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    
    dataset=pd.DataFrame(dataset)
    dataset=dataset.sample(frac =ratio,replace=True)
    dataset=dataset.values.tolist()
    
    return dataset
def bagged_chunks(my_list,new_val, n):
    #here my list is train set
    #print("mylist: ",my_list)
    #print("new_val: ",new_val)    
    #print(waitstop1)  
    #print("my_list.shape[1]: ",my_list.shape)
    #print("new_val.shape[1]: ",new_val.shape)
    list_in=list(range(0,my_list.shape[1])) # this gets random values from the number of features present.
    #print("list_in: ",list_in)    
    #random.shuffle(list_in)
    #print("list_in now: ",list_in)
    #final = [list_in[i * n:(i + 1) * n] for i in range((len(list_in) + n - 1) // n )]
    #print("final: ")    
    sample=subsample(list_in,1.00)
    
    import itertools
    list_in = list(itertools.chain.from_iterable(sample))
    
    final = [list_in[i * n:(i + 1) * n] for i in range((len(list_in) + n - 1) // n )]
   
    finalchunksublist_new_val=[]
    finalchunksublist=[]
    for li in final:
        temp=[]
        temp2=[]
        subdf=pd.DataFrame()
        temp=my_list[li].copy()
        temp2=new_val[li].copy()
             
        finalchunksublist.append(temp)
        finalchunksublist_new_val.append(temp2)
    
    print("finalchunksublist: ",finalchunksublist)
    print("finalchunksublist_new_val: ",finalchunksublist_new_val)
    #print(waitstop)    
    return finalchunksublist,finalchunksublist_new_val    
    
def chunks(my_list,new_val, n):
    print("mylist: ",my_list)
    print("my_list.shape[1]: ",my_list.shape[1])
    list_in=list(range(0,my_list.shape[1])) # this gets random values from the number of features present.
    print("list_in: ",list_in)
    
    random.shuffle(list_in)
    print("list_in now: ",list_in)
    final = [list_in[i * n:(i + 1) * n] for i in range((len(list_in) + n - 1) // n )]
    print("final: ")
    print(final)
    print(weight)
    finalchunksublist_new_val=[]
    finalchunksublist=[]
    for li in final:
        temp=[]
        temp2=[]
        subdf=pd.DataFrame()
        temp=my_list[li].copy()
        temp2=new_val[li].copy()
        #for subli in li:
        #    subdf=subdf+my_list[subli]
        #print("temp: ",temp)         
        finalchunksublist.append(temp)
        finalchunksublist_new_val.append(temp2)
    
    return finalchunksublist,finalchunksublist_new_val
        
def cascade_op(new_val_train,new_val,n,df_y_train,df_y_val,test_name):
    
    # this function only creates file for associaitive classifier nothing else
    
    #finalchunksublist,finalchunksublist_new_val=chunks(new_val_train,new_val, n)
    
    finalchunksublist,finalchunksublist_new_val=bagged_chunks(new_val_train,new_val, n)
    #print("finalchunksublist: ",finalchunksublist[1])
    #print("finalchunksublist_new_val: ",finalchunksublist_new_val[1])
    
    #cascades_classifiers()
    count=0
    for chunknum in range(len(finalchunksublist)):
        df1_train=finalchunksublist[chunknum]
        df2=df_y_train
        
        df3_train=finalchunksublist_new_val[chunknum]
        df4=df_y_val
        #print("df2before: ",df2)
        #print("df4before: ",df4)
        df1_train,df2,df3_train,df4=dataset_transformation.main(df1_train,df2,df3_train,df4) #X_transformed_perrange,df_y_train,X_transformed_perrange_val,Yval
        #print("df2bafter: ",df2)
        #print("df4after: ",df4)        
        #for train set
        Xtotal_subsample = pd.concat([df1_train,df2], axis=1)
        #Xtotal_subsample = pd.concat([df1_train,df4], axis=1)
        #print("Xtotal_subsample1: ",Xtotal_subsample)
        #print(weight22)
        #print("df1_train: ",df1_train)
        #print("df2: ",df2)
        #print("??Xtotal_subsample: ",Xtotal_subsample)
        #print(stopcheckcheck1)
        Xtest_total=Xtotal_subsample.values.tolist()         
        #print("Xtest_total: ")
        #test_name  = 'subset_datasets/' + dataset_name + '_train'+ str(var) +'.txt'
        test_name  = 'subset_datasets/' + dataset_name + '_train'+ str(count) +'.txt'
        test_name
        for var in range(len(Xtest_total)):
            
            Xtest_total[var] = [x for x in Xtest_total[var] if str(x) != 'nan' and str(x) != '']         
        
        import csv
        with open(test_name, "w",newline='\n') as f:
            writer = csv.writer(f,delimiter=' ')
            writer.writerows(Xtest_total) 
        #forval set    
        Xtotal_subsample = pd.concat([df3_train,df4], axis=1)        
        #print("Xtotal_subsample: ",Xtotal_subsample)
        #Xtotal_subsample=dataset_transformation.main(Xtotal_subsample)
        #print("Xtotal_subsample2: ",Xtotal_subsample)
        #print("df3_train: ",df3_train)
        #print("df4: ",df4)
        #print("??Xtotal_subsample: ",Xtotal_subsample)        
        
        Xtest_total=Xtotal_subsample.values.tolist()         
        #print("Xtest_total: ")
        #test_name  = 'subset_datasets/' + dataset_name + '_train'+ str(var) +'.txt'
        test_name  = 'subset_datasets/' + dataset_name + '_val'+ str(count) +'.txt'
       
        for var in range(len(Xtest_total)):
            
            Xtest_total[var] = [x for x in Xtest_total[var] if str(x) != 'nan' and str(x) != '']         
        
        import csv
        with open(test_name, "w",newline='\n') as f:
            writer = csv.writer(f,delimiter=' ')
            writer.writerows(Xtest_total)
        count+=1
        print("test_name2: ",test_name)
        print()
           
    
    #print(weight22)        
    #print(weight22)
   
def cascadestep2(new_val_train,new_val,n,df_X_val,df_X_train,df_y_train,df_y_val,test_name,dataset_name,df2_slidingwindow,df3_slidingwindow): 
    #here validation can be test set also, once the tuning of number of layers is done
    #here you need to run this code for 4 sigforests 
    num_sigforests=4 #number of sigs you want in cascade step
    new_val_train_original=new_val_train
    new_val_original=new_val
    df_y_train_original=df_y_train
    df_y_val_original=df_y_val
    flag=0
    for num in range(num_sigforests):
        #df_y_train yeh test ke lie hai
        print("number of sig forest: ",num)
        cascade_op(new_val_train_original,new_val_original,n,df_y_train_original,df_y_val_original,test_name)
        #cascade_op(new_val_train,new_val,n,df_y_train,df_y_val,test_name)
        for subsig in range(n):
            input_name  = 'subset_datasets/' + dataset_name + '_val'+ str(subsig)+'.txt'
            test_name   = 'subset_datasets/' + dataset_name + '_train' + str(subsig) + '.txt'
                        
            predictions, acc,g_rules_count,np_rules_count,hrs_acc,scores,scores_testset=sigdirect_test.main(dataset_name,subsig,input_name,test_name)
            
            #print(stophere)
            print("?????scores: ",type(scores))
            
            if subsig==0 and flag==0:
                df2=scores
                df3=scores_testset #SOME PROBLEM THERE, SOME SCORES ARE 0 FOR ALL. CHECK THIS OUT.
                flag=1
                #print("df2: ",df2)
                #print("df3: ",df3)            
                #print(weigh)
            else:
                df2 = pd.concat([df2, scores], axis=0, sort=False)
                df2=df2.groupby(level=0).mean()
                df3 = pd.concat([df3, scores_testset], axis=0, sort=False)
                df3=df3.groupby(level=0).mean()            
            print("?????@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2")
        
    #print("df2: ",df2)
    #print("df3: ",df3)
    #df2=df2.to_frame() 
    #print("????: ",type(df2))
    
    #print(type(df2),type(df_X_val))
    #print(df_X_val)
    #print(df2)
    df_y_val=df_y_val.reset_index(drop=True)
    df_X_val=df_X_val.reset_index(drop=True)
    df_y_train=df_y_train.reset_index(drop=True)
    df_X_train=df_X_train.reset_index(drop=True)        
    df2=df2.reset_index(drop=True)
    df2=df2.T
    df3=df3.reset_index(drop=True)
    df3=df3.T        
    #print(df_X_val)
    #print(df2)
    #print("df_X_val[0][0]: ",df_X_val.iloc[0])
    #print(df2.iloc[0])
    #print(weigh)
    #print(df2.shape[0],len(X_val))
    #for row in range(df2.shape[0]):
    #    new_val=pd.concat([df_X_val.iloc[row], df2.iloc[row]], axis=1, sort=False)
    #print("heredf_X_val: ",df_X_val)
    #print("??df2_slidingwindow: ",df2_slidingwindow)
    #print("df3: ",df3)
    #print("df_y_val: ",df_y_val)    
    #new_val=pd.concat([df_X_val,df2_slidingwindow, df2,df_y_val], axis=1, sort=False) # ihave to do the avg here
    new_val=pd.concat([df_X_val,df2_slidingwindow, df3,df_y_val], axis=1, sort=False) # ihave to do the avg here
    
    #print("df_X_val:df2 ",df_X_val.shape, df2.shape,new_val.shape)
    #print("new val set: ",new_val)
    #print("sig_num: ",sig_num)
    #print("heredf_X_train: ",df_X_train)
    #print("??df3_slidingwindow: ",df3_slidingwindow)
    #print("df2: ",df2)
    #print("df_y_train: ",df_y_train)    
    new_val_train= pd.concat([df_X_train,df3_slidingwindow, df2,df_y_train], axis=1, sort=False)
    #print("new_val_train: ",new_val_train)
    
    #print("??new_val: ",new_val)
    #print("new_val_train: ",new_val_train)
    #print(weighhcascade)
    #print(stopnowcheckcheck)
    return new_val_train,new_val


def multigrained_scan(X_train,X_val,X_test,window_size):
    n=window_size
    
    #print("??????? for train")
    Xtrain_sub= rolling_window(X_train, n) #Xtrain_sub is a list that has all the subsets
    #print("??????? for test")
    Xval_sub = rolling_window(X_val, n)
    
    Xtest_sub = rolling_window(X_test, n)
    
    for var in range(len(Xtrain_sub)):
        df1=pd.DataFrame(list(Xtrain_sub[var]))
        df2=pd.DataFrame(list(y_train))                      
        Xtotal_subsample = pd.concat([df1,df2], axis=1) 
        Xtest_total=Xtotal_subsample.values.tolist()
        #print("Xtest_total: ")
        #test_name  = 'subset_datasets/' + dataset_name + '_train'+str(n)+'_'+ str(var) +'.txt'
        test_name  = 'subset_datasets/' + dataset_name + '_train'+ str(var) +'.txt'
        
        for var in range(len(Xtest_total)):
            
            Xtest_total[var] = [x for x in Xtest_total[var] if str(x) != 'nan' and str(x) != '']         
        
        import csv
        with open(test_name, "w",newline='\n') as f:
            writer = csv.writer(f,delimiter=' ')
            writer.writerows(Xtest_total)
    
    for var in range(len(Xval_sub)):      
        #for test    
        df1=pd.DataFrame(list(Xval_sub[var]))
        df2=pd.DataFrame(list(y_val))                      
        Xtotal_subsample = pd.concat([df1,df2], axis=1) 
        Xtest_total=Xtotal_subsample.values.tolist()
        #print("Xtest_total: ")
        test_name  = 'subset_datasets/' + dataset_name + '_val'+ str(var) +'.txt'
        for var in range(len(Xtest_total)):
            
            Xtest_total[var] = [x for x in Xtest_total[var] if str(x) != 'nan' and str(x) != '']         
        
        import csv
        with open(test_name, "w",newline='\n') as f:
            writer = csv.writer(f,delimiter=' ')
            writer.writerows(Xtest_total)
    
    for var in range(len(Xtest_sub)):      
        #for test    
        df1=pd.DataFrame(list(Xtest_sub[var]))
        df2=pd.DataFrame(list(y_test))                      
        Xtotal_subsample = pd.concat([df1,df2], axis=1) 
        Xtest_total=Xtotal_subsample.values.tolist()
        #print("Xtest_total: ")
        test_name  = 'subset_datasets/' + dataset_name + '_test'+ str(var) +'.txt'
        for var in range(len(Xtest_total)):
            
            Xtest_total[var] = [x for x in Xtest_total[var] if str(x) != 'nan' and str(x) != '']         
        
        import csv
        with open(test_name, "w",newline='\n') as f:
            writer = csv.writer(f,delimiter=' ')
            writer.writerows(Xtest_total)
            
    #print("len(Xval_sub): ",len(Xval_sub))
    #print("len(Xtest_sub): ",len(Xtest_sub))
    #print("len(Xtrain_sub): ",len(Xtrain_sub))
    return Xtrain_sub,Xval_sub,Xtest_sub

if __name__ == "__main__":
    
    # dataSets=['iris', 'monks','transfusion','yeast','tic-tac-toe','diagnosis','facebook','horse-colic','anneal']
    dataSets = ['WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_v6_CODED']#,'iris', 'pima-indians-diabetes']
    test_count = 1
    #print(wei)
    #https://scikit-learn.org/stable/modules/preprocessing.html
    
    #for fileName in dataSets:
    #    print("Running for dataset: ", fileName)
    #50estimatorsboosting
    #led7_myversion
    #CHECK LINE 116 OF ASSOCIATIVE CLASSIFIER
    #try 'letRecog', mushroom, ionosphere,soybean,penDigits,cylBands
    fileName='adult_myversion2_nomissingdata'#'adult_myversion2_nomissingdata'#'ionosphere'#'penDigits'#'mushroomchanged'
    #'cylband_nomissingdata5'#'penDigits'#'mushroomchanged'
    #'ionosphere'#'penDigits'#'soybean_nomissingfile'#'soybean'#'ionosphere'#adult_myversion2_nomissingdata'#penDigits'#ionosphere'#letRecog'#ionosphere#adult'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_v7_10bins_coded'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_v6_CODED'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType2_FS1_SMOTE'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_30bins_v4'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_v5_30bins_transfored'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType2_FS1_SMOTE_10bins'#'WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_v5_30bins_transfored' #WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_30bins'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_v5_30bins_transfored'#'WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_v5_30bins_transfored'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_30bins'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS4_nodiscretization'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_noMissing_30bins'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_30bins'#WATT2018_12Features_Prep (1)_discretized_30bins'#WATT2018_12Features_Prep_nomissingvalues_try3_v1' #'WATT2018_12Features_Prep_nomissingvalues_numeric_weka_try3_modified' #WATT_ngap_smote_FS_f12_schema_3divnorm
    #15540 works
    dataset_name=fileName
    print("Running for dataset: ", fileName)
    import time
    t0 = time.time()
        
    num_runs=1
    total_g_rules_count=0
    total_np_rules_count=0
    acc_list=[]
    acc_list_hrs1=[]
    acc_list_hrs2=[]
    acc_list_hrs3=[]
    for num_run in range(1,num_runs+1):
        print("BEGIN THE RUN: ")
        print("How many time")
        #input_name  = 'datasets_new_originalfiles/' + dataset_name + '.txt' #train file
        input_name  = 'datasets_new_originalfiles/' + dataset_name + '_tr'+ str(num_run)+'.txt'
        test_name   = 'datasets_new_originalfiles/' + dataset_name + '_ts' + str(num_run) + '.txt'        
        nameFile="datasets_new_originalfiles" +"\\"+fileName+".names"
        #FOR TRAIN SET
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
        dfnew = dfnew.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('', np.nan)
        #now u can get the last col values that is the labels.
        y_train = dfnew.groupby(['label'] * dfnew.shape[1], 1).agg('last')
        dfnew=dfnew.where(pd.notnull(dfnew), None)
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
        X_train=pd.DataFrame(X)
        mask = X_train.applymap(lambda x: x is None)
        cols = X_train.columns[(mask).any()]
        for col in X_train[cols]:
            X_train.loc[mask[col], col] = '' 
       
        
        
        
        #SHOULD WORK FOR TEST DATASET:
        with open(test_name, 'r') as f:
            data = f.read().strip().split('\n')
        dataset = [line.strip().split(sep) for line in data]        
        df=pd.DataFrame(dataset)
        dforiginal=df
        #masking all the nones if any
        mask = df.applymap(lambda x: x is None)
        cols = df.columns[(mask).any()]
        for col in df[cols]:
            df.loc[mask[col], col] = ''
        dforiginal=df    
        dfnew=df
        dfnew = dfnew.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('', np.nan)
        #now u can get the last col values that is the labels.
        y_test=dfnew.groupby(['label'] * dfnew.shape[1], 1).agg('last')
        dfnew=dfnew.where(pd.notnull(dfnew), None)
        dfnew=dfnew.values.tolist()
        print()
        for k in range(len(dfnew)):
            dfnew[k]=[x for x in dfnew[k] if x is not None]        #removing none from list
        for i in dfnew:
            i.pop()
        X=dfnew    
        #remove none and then then pop out the last element
        #masking all the nones if any
        #print("len(X)",len(X))
        #print("len(y_test)",len(y_test))
        X_test=pd.DataFrame(X)
        y_test=pd.DataFrame(y_test)
        mask = X_test.applymap(lambda x: x is None)
        cols = X_test.columns[(mask).any()]
        for col in X_test[cols]:
            X_test.loc[mask[col], col] = ''         
        
        #END FOR MAKING TEST DATASET        
        #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle='true')
        #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=1,shuffle='true') 
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.80,shuffle='true') 
        
        
        #X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=1,shuffle='true') 
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("len(X_train) ",len(X_train))
        print("len(X_val) ",len(X_val))
        print("len(y_train) ",len(y_train))
        print("len(y_val) ",len(y_val))
        df_X_train=X_train
        df_y_train=y_train
        df_X_val=X_val
        df_y_val=y_val
        df_X_test=X_test
        df_y_test=y_test
        
        
        y_train=y_train.values
        X_train=X_train.values
        y_val=y_val.values
        X_val=X_val.values  
        X_test=X_test.values
        y_test=y_test.values
        
        #print("here: ", type(X_train))
        #print("X_train[0]: ",X_train)
        #n=10
        #x_train_subset=[]
        #Xtest_sub=[]
        windowsize_list=[5,6,7]#[14,15,16]#[5,6,7]#[14,15,16]#[5,6,7]#[8,9,10]#[8,9,10]#[5,6,7]#[8,9,10] # this code works only for three types of window sizes
        original_feature_length=X_train.shape[1]
        flag=0
        df2_slidingwindow=pd.DataFrame()
        df3_slidingwindow=pd.DataFrame()        
        df4_slidingwindow=pd.DataFrame() #for final test set 
        #print(stophere)
        #predictions, acc,g_rules_count,np_rules_count,hrs_acc,scores,scores_testset=associative_classifier_test.main(dataset_name,num_run,input_name,test_name)
        
        #print(stophere)
                
        for window_size in windowsize_list:
            print("windowwww siizzzee;;; ",window_size)
            Xtrain_sub,Xval_sub,Xtest_sub=multigrained_scan(X_train,X_val,X_test,window_size)    
            #print(Xval_sub)
            print("Xtest_sub: ",Xtest_sub)
            print("CALLING ASSOCIATIVE CLASSIFIER")
            #print(weigh)
            #df1=pd.Series([])
            print("len(Xval_sub): ",len(Xval_sub))
            print("len(Xtrain_sub): ",len(Xtrain_sub))
            
            
            for sig_num in range(len(Xval_sub)):
                print("RUN NUMBER INSIDE HERE: ",sig_num)
                #print(weigh)
                #val/test
                input_name  = 'subset_datasets/' + dataset_name + '_val'+ str(sig_num)+'.txt'
                test_name   = 'subset_datasets/' + dataset_name + '_train' + str(sig_num) + '.txt'
                                
                predictions, acc,g_rules_count,np_rules_count,hrs_acc,scores,scores_testset=sigdirect_test.main(dataset_name,sig_num,input_name,test_name)
                #print(stop)
                    
                #print(stop)
                input_name  = 'subset_datasets/' + dataset_name + '_val'+ str(sig_num)+'.txt'
                test_name   = 'subset_datasets/' + dataset_name + '_test' + str(sig_num) + '.txt'
                
                predictions, acc,g_rules_count,np_rules_count,hrs_acc,scores_testingphase,scores_actualtestset=sigdirect_test.main(dataset_name,sig_num,input_name,test_name)
                print("?????scores: ",scores)
                print("?????scores: ",type(scores))
                #print('total time:',int(time.time()-t0))
                #print(stopp)            
                #print(stopp)    
                                        
                    
                if sig_num==0:
                    df2=scores
                    df3=scores_testset
                    df4=scores_actualtestset
                                 
                else:
                    df2 = pd.concat([df2, scores], axis=0, sort=False)
                    df3 = pd.concat([df3, scores_testset], axis=0, sort=False)
                    df4 = pd.concat([df4, scores_actualtestset], axis=0, sort=False)
                print("?????@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2")
            
                
            #print("df2: ",df2)
            #print("df3: ",df3)
            #df2=df2.to_frame() 
            #print("????: ",type(df2))
            #print(weigh)
            #print(type(df2),type(df_X_val))
            #print(df_X_val)
            #print(df2)
            #print(stop)            
            df_y_val=df_y_val.reset_index(drop=True)
            df_X_val=df_X_val.reset_index(drop=True)
            
            df_X_test=df_X_test.reset_index(drop=True)
            df_y_test=df_y_test.reset_index(drop=True)
            
            df_y_train=df_y_train.reset_index(drop=True)
            df_X_train=df_X_train.reset_index(drop=True)        
            df2=df2.reset_index(drop=True)
            df2=df2.T
            df3=df3.reset_index(drop=True)
            df3=df3.T        
            df4=df4.reset_index(drop=True)
            df4=df4.T        
            
            #print(df_X_val)
            #print(df2)
            #print("df_X_val[0][0]: ",df_X_val.iloc[0])
            #print(df2.iloc[0])
            #print("**1***************8 : ")
            #print("**2 : ",df2)
            #print("**3 : ",df3)
            #print("**4 : ",df4)
            #print(weigh)
            #print(df2.shape[0],len(X_val))
            #for row in range(df2.shape[0]):
            #    new_val=pd.concat([df_X_val.iloc[row], df2.iloc[row]], axis=1, sort=False)
            if flag==0:
                new_val=pd.concat([df_X_val, df2], axis=1, sort=False)
                #print("df_X_val:df2 ",df_X_val.shape, df2.shape,new_val.shape)
                #print("new val set: ",new_val)
                #print("sig_num: ",sig_num)
                
                new_val_train= pd.concat([df_X_train, df3], axis=1, sort=False)
                new_test = pd.concat([df_X_test, df4], axis=1, sort=False)
                #print("######################################################new_val_train1: ",new_val_train)
                flag+=1
            elif flag>0 and flag<2:
                new_val=pd.concat([new_val, df2], axis=1, sort=False)
                #print("df_X_val:df2 ",df_X_val.shape, df2.shape,new_val.shape)
                #print("new val set: ",new_val)
                #print("sig_num: ",sig_num)                
                new_val_train= pd.concat([new_val_train, df3], axis=1, sort=False)
                new_test = pd.concat([new_test, df4], axis=1, sort=False)
                #print("new_val_train1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: ",new_val_train)
                flag+=1
            else:
                new_val=pd.concat([new_val, df2,df_y_val], axis=1, sort=False)
                #print("df_X_val:df2 ",df_X_val.shape, df2.shape,new_val.shape)
                #print("new val set: ",new_val)
                #print("sig_num: ",sig_num)                
                new_val_train= pd.concat([new_val_train, df3,df_y_train], axis=1, sort=False)
                new_test= pd.concat([new_test, df4,df_y_test], axis=1, sort=False)
                
                #print("new_val_train1: ",new_val_train)                
                
            sizeold=new_val_train.shape
            #keep df2 and df3 as they are the features from 5 sigs from rolling window and u need to append them
            #df2_slidingwindow[window_size]=df2
            #df3_slidingwindow[window_size]=df3
            #df2_slidingwindow = df2 
            df2_slidingwindow = pd.concat([df2_slidingwindow, df2], axis=1, sort=False)
            df3_slidingwindow = pd.concat([df3_slidingwindow, df3], axis=1, sort=False)
            df4_slidingwindow = pd.concat([df4_slidingwindow, df4], axis=1, sort=False)
        
        '''    
        print("new_val.shape: ",new_val)
        print("new_val_train.shape: ",new_val_train)
        print("SHAPES1: ", new_val.shape,new_val_train.shape)
        print("df2_slidingwindow: ",df2_slidingwindow)
        print("df3_slidingwindow: ",df3_slidingwindow)
        print("df4: ",df4)
        print("df4_slidingwindow: ",df4_slidingwindow)
        #print(weighstop)
        '''
        gc.collect(generation=2)
        print('Garbage Collector')
        sys.stdout.flush() 
        new_val_train_original=new_val_train
        df_y_train_original=df_y_train
        new_val_original=new_val
        df_y_val_original=df_y_val  
        df_y_test_original=df_y_test
        n=7 # here n is the number of random samples you generate. most probably check again
        
                
        flagstage=1 #flagestaage is 1 if validation phase is going on and 0 if test phase is going on.
        #data transformation of 
        num_layers=5
        #making new validation set for cascade structure: CASCADE SET
        
        for layer in range(num_layers): #FOR NOW LETS SAY WE HAVE JUST ONE LAYER
        #for train set
            #dataset_transformation_mgstage1
            print("before new_val_train: ",new_val_train)
            new_val_train,df_y_train=dataset_transformation_mgstage1.main(new_val_train,original_feature_length,df_y_train)
            print("after new_val_train: ",new_val_train)
            #print(checkthisone)
            new_val,df_y_val=dataset_transformation_mgstage1.main(new_val,original_feature_length,df_y_val)
            
            #new_val_train=dataset_transformation.main(new_val_train,new_val)
            #new_val=dataset_transformation.main(new_val,original_feature_length)            
            #print("SHAPES1: ", new_val.shape,new_val_train.shape)
            #print(weighhhh)
            #print("windowwww siizzzee inside;;; ",n)
            #print("new_val_train: ",new_val_train)
            #print("new_val: ",new_val)
            # just printing into the file for now to check the output
            #testing  =  'subset_datasets/' +  dataset_name + '_testtry'+'.csv'
            #new_val_train.to_csv(testing)
            #testing  =  'subset_datasets/' +  dataset_name + '_testtryval'+'.csv'
            #new_val.to_csv(testing)
            #print(weigh)
            #problem is tthat the values are different. because you are changing the names. so train has lot more rows and then its values they are changed.
            #even the last col of one set in range of 3k while for the other it is in 1k so not consistant.
            #same feature name also in each dataset can correspond to different feature value in the other dataset.            
            #print(stophere3)            
            new_val_train,new_val,lastlayer_pred_list=cascadestep2_valstep(new_val_train,new_val,n,df_X_val,df_X_train,df_y_train,df_y_val,test_name,dataset_name,df2_slidingwindow,df3_slidingwindow,layer,num_layers,flagstage)
            #print("new_val_train232: ")
            #print(new_val_train)
            #print("new_val232: ")
            #print(new_val)
            #print("sizeold: ",sizeold)
            #print("sizenew: ",new_val_train.shape)  
            
        
        
        #print("df4_slidingwindow: ",df4_slidingwindow)
        print("FOCUSSSSSSS TEST SET NOW BEGINS: ")
        df_y_train=df_y_train_original
        new_val_train=new_val_train_original
        df_y_val=df_y_val_original
        new_val=new_val_original     
        lastlayer_pred_list=[]
        flagstage=0
        #FOR TEST SET BASED ON NUMBER OF LAYERS RECEIVED FROM PREVIOUS SET
        for layer in range(num_layers): #FOR NOW LETS SAY WE HAVE JUST ONE LAYER
        #for train set
            #dataset_transformation_mgstage1
            new_test,df_y_test=dataset_transformation_mgstage1.main(new_test,original_feature_length,df_y_test)
            #print(checkthisone)
            new_val,df_y_val=dataset_transformation_mgstage1.main(new_val,original_feature_length,df_y_val)
            
            #print(stophere3)            
            new_test,new_val,lastlayer_pred_list=cascadestep2_valstep(new_test,new_val,n,df_X_val,df_X_test,df_y_test,df_y_val,test_name,dataset_name,df2_slidingwindow,df4_slidingwindow,layer,num_layers,flagstage)
            #print("new_val_train232: ")
            #print(new_test)
            #print("new_val232: ")
            #print(new_val)
            #print("sizeold: ",sizeold)
            #print("sizenew: ",new_test.shape)  
        
        print("COMPLETED ALL THE LAYERS FOR TESTING SET*****")
        #print("lastlayer_pred_list: ",lastlayer_pred_list)
        lastlayer_pred_list = pd.DataFrame(lastlayer_pred_list)
        lastlayer_pred_list = lastlayer_pred_list.transpose()
        #print("df version lastlayer_pred_list: ",lastlayer_pred_list)
        import sklearn
        from sklearn import preprocessing
        enc = sklearn.preprocessing.OrdinalEncoder()
        enc.fit(lastlayer_pred_list)
        lastlayer_pred_list=enc.transform(lastlayer_pred_list)  
        #print("after encoding : lastlayer_pred_list: ",lastlayer_pred_list)
        lastlayer_pred_list=pd.DataFrame(lastlayer_pred_list)     
        #print("dataframe:  lastlayer_pred_list: ",lastlayer_pred_list)
        finalpred_df=lastlayer_pred_list.mode(axis=1)[0]
        #print("lastlayer_pred_list.mode(axis=1): ",lastlayer_pred_list.mode(axis=1))        
        #print("lastlayer_pred_list.mode(axis=1)[0]: ",lastlayer_pred_list.mode(axis=1)[0])
        #print("ytest originally df_y_test_original: ",df_y_test_original)
        enc.fit(df_y_test_original)
        df_y_test_original=enc.transform(df_y_test_original)          
        #print("after encoding test set: ",df_y_test_original)
        Ytest=df_y_test_original.astype(int)
        ypred_final=finalpred_df.astype(int)
        #print(stopletsseeeweigh)
        #finalpred_df=finalpred_df.values.tolist()
        #df_y_test_original=df_y_test_original.values.tolist()
        count=0
        #print(len(finalpred_df))
        i=0
        #print("*********",ypred_final[i],Ytest[i])
        #print("1*********",ypred_final[i])
        #print(Ytest[0][i])
        for i in range(len(finalpred_df)): #finalpred_df.shape[1]
            if(ypred_final[i]==Ytest[i]):
                #print("*****34567*",ypred_final[i],Ytest[i])
                count=count+1
        print("count: ",count)
        print("Final accuracy of test set is : ", (count/len(ypred_final)))
        acc_list.append(count/len(ypred_final))
        #print(stopletsseeeweigh)
        #print(stop1)
        #print(weighh)
        #predict()
        #print(stop)
        #now call test set with these many layers which give the final best performance:
        #perform multigrain step as well as cascade with the number of layers determined before
        #change cascade and divide sig into some parts.. then use the scores from each sig bin avg them and add it 
                      
        
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@2~~~~~~~~~~~~~~~~~~~~~~~~~~~NEXT RUN: ",num_run)
        print(stopp) 
    print("acc_list: ",acc_list)
    print('total time:',int(time.time()-t0))
                
    print(stopp)    
        
         
    
        
        
            
            

        


