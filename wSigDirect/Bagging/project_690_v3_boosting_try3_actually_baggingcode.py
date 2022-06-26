# final code for bagging
import random
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from random import randrange
from sklearn.model_selection import StratifiedKFold
import miniproject as ml
from sklearn.utils import resample
from random import choices
from sklearn.model_selection import KFold
from sklearn.utils.multiclass import type_of_target
import csv
import miniproject as ml
import statistics
# import Logistic as NN
# from src.NeuralNetwork import NeuralNetwork
# import src.utils as utils
import neuralNW
import associative_classifier_test
import preprocessing
import warnings
import pandas as pd
import random
from random import randint
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

def bagging_func(y_bagging, len_pred, Ytest):
    print("Hi")
    ypred_bagging=[]
    for j in range(0,len_pred):
        temp=[]
        for i in range(len(y_bagging)):
            temp.append(y_bagging[i][j])
        ypred_bagging.append(temp)
    
    print("!!!!!!!!!! ypred_bagging !!!!!!!!!!!!") 
    print(ypred_bagging)
    ypred_final=[]
    for i in range(len(ypred_bagging)):
        #print("len_pred: 3")
        #print(len(set(ypred_bagging[i])))        
        if len(set(ypred_bagging[i])) == 4: 
            print( "No mode found" )
            print("setting value: statistics.mode(ypred_bagging[i]): ",ypred_bagging[i][0])
            ypred_final.append(ypred_bagging[i][0])
        else:
            #print("statistics.mode(ypred_bagging[i]): ",statistics.mode(ypred_bagging[i]))
            ypred_final.append(statistics.mode(ypred_bagging[i]))
    print("ypred_final")
    print(ypred_final)
    print("len(ypred_final): ",type(ypred_final))
    print("Ytest")
    print(Ytest)    
    print("len(Ytest): ",type(Ytest))
    count=0
    for i in range(len(ypred_final)):
        if(ypred_final[i]==Ytest[i]):
            count=count+1
    print("count: ",count)
    print("bagging accuracy: ", (count/len(ypred_final)))
    return ypred_final

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
    print("#######temp:")
    print(temp)
    return sample


def merge(list1, list2): 
      
    merged_list = list(zip(list1, list2))
    
    
    return merged_list

def update_dataset(weight_dict,X_train, X_test, y_train, y_test):
    print("update_dataset")
    
    test_name  = 'datasets_new_boosting/' + dataset_name + '_testoriginal'+ str(num_run) +'.txt' 
    with open(test_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dxy_test = [line.strip().split(sep) for line in data]
    
    train_name  = 'datasets_new_boosting/' + dataset_name + '_trainoriginal'+ str(num_run) +'.txt' 
    with open(train_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dxy_train = [line.strip().split(sep) for line in data] 
    
    for i in range(len(dxy_test)):
        temp=[]
        print("ORIGINAL dxy_test[i]: ",dxy_test[i])
        for values in (dxy_test[i]): 
            print("values: ",values)
            
            values=(float(values)*(weight_dict[values]))
            temp.append(values)
        dxy_test[i]=temp  
        print("AFTER dxy_test[i]: ",dxy_test[i])
    #print("temp: ",temp)
    
    X_test=dxy_test
    print("1FOCUS HERE X_train: ",X_train)
    for i in range(len(dxy_train)):
        temp=[]
        #print("ORIGINAL dxy_train[i]: ",dxy_train[i])
        for values in (dxy_train[i]): 
            #print("values: ",values)
            
            values=(float(values)*(weight_dict[values]))
            temp.append(values)
        dxy_train[i]=temp  
        #print("AFTER dxy_train[i]: ",dxy_train[i])
       
    
    X_train=dxy_train
    
    #print("2FOCUS HERE X_train: ",X_train)
    #print(weigh)
     
    '''
    for i in range(len(dxy_train)):        
        for values in (dxy_train[i]):            
            values=(float(values)*(weight_dict[values]))
            
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):            
            X_train[i][j]=(float(X_train[i][j])*(weight_dict[X_train[i][j]]))    
    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):            
            X_test[i][j]=(float(X_test[i][j])*(weight_dict[X_test[i][j]]))
     
    #for i in range(y_train_updated.shape[0]):
    #    y_train_updated[i]=(float(y_train_updated[i])*(weight_dict[y_train_updated[i]]))
        
    #for i in range(y_test_updated.shape[0]):
    #    y_test_updated[i]=(float(y_test_updated[i])*(weight_dict[y_test_updated[i]]))    
     '''     
    print("AUR YAHAN>>> X_test_original: ",X_test_original)
    return weight_dict,X_train, X_test, y_train, y_test        
            

def boosting_func(predictions,X_train, X_test_updated, y_train, y_test, sample_weight,weight_dict,X_test):
    print("Begin Boosting: ")
    #print("predictions: ",predictions)
    #print("y_test: ",y_test)
    incorrect = predictions != y_test
    #print("incorrect: ")
    #print(incorrect)
           
    #print("here X_test_updated: ",X_test_updated)
    estimator_error=0
    test_name  = 'datasets_new_boosting/' + dataset_name + '_testoriginal'+ str(num_run) +'.txt' 
    with open(test_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dxy = [line.strip().split(sep) for line in data]
    
    #print("data: ",dxy)   
    
    #print("weight_dict: ",weight_dict)
    for i in range(incorrect.shape[0]):
        #print("?dxy[i]: ",dxy[i])
                
        Xtest_feature=list(set(dxy[i]))
        
        #print("?Xtest_feature: ",Xtest_feature)
        for feature in Xtest_feature:
            #weight_dict[feature]=weight_dict[feature]*np.exp(-(alpha)* (-1))
            #print("feature: ",feature)
            #print("weight_dict[feature]: ",weight_dict[feature])
            #print("incorrect[i]: ",incorrect[i])
            estimator_error = estimator_error+ float(incorrect[i])*weight_dict[feature]
        
    
    print("estimator_error:")  
    print(estimator_error)
    # this hsould not be smaple weight, but the weight of all the train samples so change it
    sum_val=sum(weight_dict.values())
    #misclassification_rate = estimator_error / sum_val  
    #estimator_error=misclassification_rate
    
    
    n_classes_=3 #FOR IRIS ONLY
    # if worse than random guess, stop boosting
    #if estimator_error.all() >= 1.0 - 1 / n_classes_: # you need to change this
    #    return None 
    print("estimator_error: ",estimator_error)
    alpha=1/2 *(np.log((1-estimator_error)/estimator_error))
    #sample_weight=sample_weight+1
    print("np.log((1-estimator_error)/estimator_error): ",np.log((1-estimator_error)/estimator_error))
    print("alpha: ",alpha)
    for i in range(len(incorrect)):
        
        if incorrect[i]==True:
            #sample_weight=np.exp(-(alpha)* (-1))
            Xtest_feature=list(set(dxy[i]))
            for feature in Xtest_feature:
                weight_dict[feature]=weight_dict[feature]*np.exp(-(alpha)* (-1))
            
            #weight_dict = dict.fromkeys(feature_values, sample_weight)
        else:
            #sample_weight=sample_weight*np.exp(-(alpha)* (1))
            Xtest_feature=list(set(dxy[i]))
            for feature in Xtest_feature:
                weight_dict[feature]=weight_dict[feature]*np.exp(-(alpha)* (1))            
    
    
    return weight_dict

        



if __name__ == "__main__":
    
    # dataSets=['iris', 'monks','transfusion','yeast','tic-tac-toe','diagnosis','facebook','horse-colic','anneal']
    dataSets = ['hepati']#,'iris', 'pima-indians-diabetes']
    test_count = 1
    
    #for fileName in dataSets:
    #    print("Running for dataset: ", fileName)

    fileName='hepati'
    dataset_name=fileName
    print("Running for dataset: ", fileName)
    
    num_runs=1
    for num_run in range(num_runs):
        print("How many time")
        input_name  = 'datasets_new_originalfiles/' + dataset_name + '_tr'+ str(num_runs) +'.txt' #train file
        
        sep = ' '
        with open(input_name, 'r') as f:
            data = f.read().strip().split('\n')
        dataset = [line.strip().split(sep) for line in data]        
        
       
        df=pd.DataFrame(dataset)
        
        #masking all the nones if any
        mask = df.applymap(lambda x: x is None)
        cols = df.columns[(mask).any()]
        for col in df[cols]:
            df.loc[mask[col], col] = ''
        
        
        
        
              
        Y = df.iloc[:,-1].values # Here first : means fetch all rows :-1 means except last column
        X = df.iloc[:,:-1].values # : is fetch all rows 3 means 3rd column  
        
        print("len(X)",len(X))
        print("len(Y)",len(Y))
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle='true')
        print("len(X_train): ",len(X_train))
        print("len(X_test): ",len(X_test))
        print("len(y_train): ",len(y_train))
        print("len(y_test): ",len(y_test))
        X_test_original=X_test
        df3_test=pd.DataFrame(list(X_test_original))              
        test_name  = 'datasets_new_boosting/' + dataset_name + '_testoriginal'+ str(num_run) +'.txt'
        df3_test.to_csv(test_name,sep=' ',index=False,header=False) 
        
        
        df3_test=pd.DataFrame(list(X_train))              
        test_name  = 'datasets_new_boosting/' + dataset_name + '_trainoriginal'+ str(num_run) +'.txt'
        df3_test.to_csv(test_name,sep=' ',index=False,header=False)         
        
        df1=pd.DataFrame(list(X_train))
        df2=pd.DataFrame(list(y_train))        
        Xtotal_subsample = pd.concat([df1,df2], axis=1)
        
        counter=1
        
        y_bagging=[]
        y_boosting=[]
        weight_dict={}
        print(type(X_train))
        #feature_values=list(set().union(X_train, X_test, y_train, y_test))
        feature_values=list(set().union(X_train.flatten(), X_test.flatten(), y_train.flatten(), y_test.flatten()))
        
        n=len(feature_values)
        
        #sample_weight=1/n
        sample_weight=1/X_train.shape[0]
        weight_dict = dict.fromkeys(feature_values, sample_weight)
        print("weight_dict: ")
        print(weight_dict)
        num_estimators=4
        
        X_train_updated = X_train
        X_test_updated = X_test
        y_train_updated = y_train
        y_test_updated = y_test
        y_boosting=[]
            
        for counter in range(1,num_estimators):
            print("1??????X_train_updated :",X_train_updated)
            print("1??????X_train:",X_train)            
            weight_dict,X_train_updated, X_test_updated, y_train_updated, y_test_updated = update_dataset(weight_dict,X_train, X_test, y_train, y_test)
            print("2??????X_train_updated: ",X_train_updated)            
            print("2??????X_train",X_train)
            #print(weigh)
            #check this out kaunsa dataset aega
            df1=pd.DataFrame(list(X_train_updated))
            df2=pd.DataFrame(list(y_train_updated))        
            Xtotal_subsample = pd.concat([df1,df2], axis=1)            
            Xtotal_subsample_list=Xtotal_subsample.values.tolist()
            
            random.shuffle(Xtotal_subsample_list)
            sample=subsample(Xtotal_subsample_list, 0.60)
                       
            Xtotal=pd.DataFrame(list(sample))
               
            train_name  = 'datasets_new_boosting/' + dataset_name + '_train'+ str(counter) +'.txt'
            #df.to_csv(train_name, sep='\t', encoding='utf-8')
            Xtotal.to_csv(train_name,sep=' ',index=False,header=False)
            #creating test set: 
            
            
            df1_test=pd.DataFrame(list(X_test_updated))
            df2_test=pd.DataFrame(list(y_test_updated))   
            Xtest_total = pd.concat([df1_test,df2_test], axis=1)
            
                
            
            test_name  = 'datasets_new_boosting/' + dataset_name + '_test'+ str(counter) +'.txt'
            #df.to_csv(train_name, sep='\t', encoding='utf-8')
            Xtest_total.to_csv(test_name,sep=' ',index=False,header=False)    
              
            predictions=associative_classifier_test.main(dataset_name,counter)
            
            weight_dict=boosting_func(predictions,X_train_updated, X_test_updated, y_train_updated, y_test_updated, sample_weight,weight_dict,X_test_original)
                        
            y_boosting.append(predictions)
            
            #y_bagging.append(predictions)
        
        print("^^^^^^^^^^^^^^^^^^^^^y_boosting: ",y_boosting)
        print("DONE")
        bagging_func(y_boosting,len(X_test),y_test)                
        #bagging_func(y_bagging,len(X_test),y_test)    
        
        
        
        
        
        
            
            
            

        


