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
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
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

    fileName='hepati'
    dataset_name=fileName
    print("Running for dataset: ", fileName)
    
    num_runs=2
    for num_run in range(1,num_runs):
        
        input_name  = 'datasets_new_originalfiles/' + dataset_name + '_tr'+ str(1) +'.txt' #train file
        #test_name   = 'uci/' + dataset_name + '_ts' + str(1) + '.txt' #test file  
        
        sep = ' '
        with open(input_name, 'r') as f:
            data = f.read().strip().split('\n')
        dataset = [line.strip().split(sep) for line in data]        
        
        
        #random.shuffle(dataset)
        print("dataset: ")
        print(dataset)
        print("DONE")
        '''
        features_int = set(x for line in dataset for x in line[:-1])
        columns = [str(x) for x in features_int] + ['class']
        
        row_dicts = list(map(lambda row: {**{k:1 for k in set(row[:-1])}, **{'class':int(row[-1])}}, dataset))
        df = pd.DataFrame(row_dicts, columns=columns)
        df = df.fillna(0).astype(int) 
        '''
        df=pd.DataFrame(dataset)
        print("df: ")
        print(df)
        
        #masking all the nones if any
        mask = df.applymap(lambda x: x is None)
        cols = df.columns[(mask).any()]
        for col in df[cols]:
            df.loc[mask[col], col] = ''
        print("DF again: ")
        print(df)
        
        
        
              
        Y = df.iloc[:,-1].values # Here first : means fetch all rows :-1 means except last column
        X = df.iloc[:,:-1].values # : is fetch all rows 3 means 3rd column  
        print("X: ")
        print(X)
        print("Y: ")
        print(Y)
        print("len(X)",len(X))
        print("len(Y)",len(Y))
        print("@##@#@: ",type(X))
        print(type(Y))
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle='true')
        print("len(X_train): ",len(X_train))
        print("len(X_test): ",len(X_test))
        print("len(y_train): ",len(y_train))
        print("len(y_test): ",len(y_test))
        #print(type_of_target(Y))
        
        '''
        #creating test set: 
        Xtest_total=merge(X_test, y_test)
        Xtest_total=pd.DataFrame(list(Xtest_total))
        test_name  = 'datasets_new/' + dataset_name + '_test'+ str(counter) +'.txt'
        #df.to_csv(train_name, sep='\t', encoding='utf-8')
        df.to_csv(test_name,index=False,header=False)    
        '''
        
        #skf = KFold(n_splits=3, shuffle=True)
        #X= str(X_train[:][0])[1 : -1]
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~X",X)
        #Y=str(y_train)
        Xtotal_subsample=merge(X_train, y_train)
        print(Xtotal_subsample)  
        print(type(Xtotal_subsample))
        print(Xtotal)
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        counter=1
        #for train_index, test_index in skf.split(X_train, y_train):
        for counter in range(1,3):
            
            
            sample1=subsample(Xtotal_subsample, 0.80)
            print("sample1:")
            print(sample1)            
            Xtotal=pd.DataFrame(list(sample1))
            print(Xtotal)   
            train_name  = 'datasets_new/' + dataset_name + '_train'+ str(counter) +'.txt'
            #df.to_csv(train_name, sep='\t', encoding='utf-8')
            Xtotal.to_csv(train_name,index=False,header=False)
            #creating test set: 
            #X= str(X_test[i][0])[1 : -1]
            Xtest_total=merge(X_test, y_test)
            Xtest_total=pd.DataFrame(list(Xtest_total))
            test_name  = 'datasets_new/' + dataset_name + '_test'+ str(counter) +'.txt'
            #df.to_csv(train_name, sep='\t', encoding='utf-8')
            Xtest_total.to_csv(test_name,index=False,header=False)    
                
            associative_classifier_test.main(dataset_name)
                        
            
        
        
        
        
        
        
            
            
            

        


