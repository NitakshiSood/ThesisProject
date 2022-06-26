# just for test
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
    #print("INSIDE FUNDCTION")
    #print(merged_list)
    
    return merged_list 

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
        print("len_pred: 3")
        print(len(set(ypred_bagging[i])))        
        if len(set(ypred_bagging[i])) == 3: 
            print( "No mode found" )
            print("setting value: statistics.mode(ypred_bagging[i]): ",ypred_bagging[i][0])
            ypred_final.append(ypred_bagging[i][0])
        else:
            print("statistics.mode(ypred_bagging[i]): ",statistics.mode(ypred_bagging[i]))
            ypred_final.append(statistics.mode(ypred_bagging[i]))
    print("ypred_final")
    print(ypred_final)
    print("len(ypred_final): ",len(ypred_final))
    print("len(Ytest): ",len(Ytest))
    count=0
    for i in range(len(ypred_final)):
        if(ypred_final[i]==Ytest[i]):
            count=count+1
    print("count: ",count)
    print("bagging accuracy: ", (count/len(ypred_final)))
    return ypred_final


if __name__ == "__main__":
    
    # dataSets=['iris', 'monks','transfusion','yeast','tic-tac-toe','diagnosis','facebook','horse-colic','anneal']
    dataSets = ['iris']#,'iris', 'pima-indians-diabetes']
    test_count = 1
    
    #for fileName in dataSets:
    #    print("Running for dataset: ", fileName)

    fileName='iris'
    dataset_name=fileName
    print("Running for dataset: ", fileName)
    
    num_runs=2
    for num_run in range(1,num_runs):
        
        input_name  = 'datasets_new_originalfiles/' + dataset_name + '_tr'+ str(num_runs) +'.txt' #train file
        
        sep = ' '
        with open(input_name, 'r') as f:
            data = f.read().strip().split('\n')
        dataset = [line.strip().split(sep) for line in data]        
        
        
        #random.shuffle(dataset)
        #print("dataset: ")
        #print(dataset)
        #print("DONE")

        df=pd.DataFrame(dataset)
        #print("df: ")
        #print(df)
        
        #masking all the nones if any
        mask = df.applymap(lambda x: x is None)
        cols = df.columns[(mask).any()]
        for col in df[cols]:
            df.loc[mask[col], col] = ''
        #print("DF again: ")
        #print(df)
        
        
        
              
        Y = df.iloc[:,-1].values # Here first : means fetch all rows :-1 means except last column
        X = df.iloc[:,:-1].values # : is fetch all rows 3 means 3rd column  
        #print("X: ")
        #print(X)
        #print("Y: ")
        #print(Y)
        print("len(X)",len(X))
        print("len(Y)",len(Y))
        #print("@##@#@: ",type(X))
        #print(type(Y))
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle='true')
        print("len(X_train): ",len(X_train))
        print("len(X_test): ",len(X_test))
        print("len(y_train): ",len(y_train))
        print("len(y_test): ",len(y_test))
        
        df1=pd.DataFrame(list(X_train))
        df2=pd.DataFrame(list(y_train))        
        Xtotal_subsample = pd.concat([df1,df2], axis=1)
        print("#########################df1")
        #print(Xtotal_subsample)
        
        counter=1
        
        y_bagging=[]
        for counter in range(1,3):
                                    
            Xtotal_subsample_list=Xtotal_subsample.values.tolist()
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
            #print(Xtotal_subsample_list)
            random.shuffle(Xtotal_subsample_list)
            sample=subsample(Xtotal_subsample_list, 0.60)
            print("sample: ",counter)
            print(sample)            
            Xtotal=pd.DataFrame(list(sample))
            
            
            train_name  = 'datasets_new/' + dataset_name + '_train'+ str(counter) +'.txt'
            
            Xtotal.to_csv(train_name,sep=' ',index=False,header=False)
            df1_test=pd.DataFrame(list(X_test))
            df2_test=pd.DataFrame(list(y_test))   
            Xtest_total = pd.concat([df1_test,df2_test], axis=1)
            test_name  = 'datasets_new/' + dataset_name + '_test'+ str(counter) +'.txt'
            
            Xtest_total.to_csv(test_name,sep=' ',index=False,header=False) 
            
            
            '''
            predictions=associative_classifier_test.main(dataset_name,counter)                
            
            y_bagging.append(predictions)
                        
        bagging_func(y_bagging,len(X_test),y_test)            
         '''               
            
        
        
        
        
            
            
            

        


