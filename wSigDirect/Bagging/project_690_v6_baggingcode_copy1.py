# very final code for bagging, experiments done on this version.
#WORKS FOR BAGGING ON NEW SIGDIRECT CODE
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
from scipy.stats import mode
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

def bagging_func(y_bagging, len_pred, Ytest):
    #print("Hi")
    ypred_bagging=[]
    test_name  =   'datasets_bagging/' + dataset_name + '_ytestoriginal'+ str(num_run) +'.txt'
    with open(test_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dy_test = [line.strip().split(sep) for line in data]
    
    
    #y_test=dy_test
    Ytest=pd.DataFrame(dy_test)     
    for j in range(0,len_pred):
        temp=[]
        for i in range(len(y_bagging)):
            temp.append(y_bagging[i][j])
        ypred_bagging.append(temp)
    
    #print("!!!!!!!!!! ypred_bagging !!!!!!!!!!!!") 
    #print(ypred_bagging)
    ypred_final=[]
    for i in range(len(ypred_bagging)):
        #print("len_pred: 3")
        #print(ypred_bagging[i]) 
        #print("ypred_bagging[i]: max method: ",max(set(ypred_bagging[i]), key=ypred_bagging[i].count))
        #print("ypred_bagging[i]: mode: ",mode(ypred_bagging[i]))
        mode_val=max(set(ypred_bagging[i]), key=ypred_bagging[i].count)
        ypred_final.append(mode_val)
        '''
        if len(set(ypred_bagging[i])) == 4: 
            print( "No mode found" )
            print("setting value: statistics.mode(ypred_bagging[i]): ",ypred_bagging[i][0])
            ypred_final.append(ypred_bagging[i][0])
        else:
            #print("statistics.mode(ypred_bagging[i]): ",statistics.mode(ypred_bagging[i]))
            ypred_final.append(statistics.mode(ypred_bagging[i]))
            '''
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
    print("count: ",count)
    print("bagging accuracy: ", (count/len(ypred_final)))
    return (count/len(ypred_final))# ypred_final

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    
    dataset=pd.DataFrame(dataset)
    dataset=dataset.sample(frac =ratio,replace=True)
    dataset=dataset.values.tolist()
    #print("type of dataset: ",type(dataset))
    #print("dataset: ")
    #print(dataset)
    #print(weigh)
    return dataset

   

def merge(list1, list2): 
      
    merged_list = list(zip(list1, list2))
    
    
    return merged_list
def update_dataset(weight_dict,X_train, X_test, y_train, y_test):
    #print("update_dataset")
     
    #old version with list
    test_name  =   'datasets_bagging/' + dataset_name + '_testoriginal'+ str(num_run) +'.txt' 
    with open(test_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dxy_test = [line.strip().split(sep) for line in data]
    
    train_name  =  'datasets_bagging/' +  dataset_name + '_trainoriginal'+ str(num_run) +'.txt' 
    with open(train_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dxy_train = [line.strip().split(sep) for line in data] 
    
    #print(type(dxy_test))
    #print(type(dxy_train))
    
    
    test_name  =   'datasets_bagging/' + dataset_name + '_ytestoriginal'+ str(num_run) +'.txt'
    with open(test_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dy_test = [line.strip().split(sep) for line in data]
    
    train_name  =   'datasets_bagging/' + dataset_name + '_ytrainoriginal'+ str(num_run) +'.txt' 
    with open(train_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dy_train = [line.strip().split(sep) for line in data]    
    
    y_test=pd.DataFrame(dy_test)
    y_train=pd.DataFrame(dy_train)
    
    #print("NOW FOCUSSSSSSS---------: ")
    for i in range(len(dxy_test)):
        temp=[]
        #print("ORIGINAL dxy_test[i]: ",dxy_test[i])
        for values in (dxy_test[i]):             
            #print("1values: ",values)
            #print("weight_dict[values]: ",weight_dict[values])
            values=(float(values)*(weight_dict[values]))
            #print("2values: ",values)
            temp.append(values)
            
        dxy_test[i]=temp
        
        #print("AFTER dxy_test[i]: ",dxy_test[i])
    #print("temp: ",temp)
    
    #X_test=dxy_test
    X_test=pd.DataFrame(dxy_test)
    #print("1FOCUS HERE X_train: ",X_train)
    for i in range(len(dxy_train)):
        temp=[]
        #print("ORIGINAL dxy_train[i]: ",dxy_train[i])
        for values in (dxy_train[i]): 
         #   print("values: ",values)
            values=(float(values)*(weight_dict[values]))
            temp.append(values)
        dxy_train[i]=temp  
        #print("AFTER dxy_train[i]: ",dxy_train[i])        
     
     
    X_train=pd.DataFrame(dxy_train)
    
    return weight_dict,X_train, X_test, y_train, y_test
      

if __name__ == "__main__":
    
    # dataSets=['iris', 'monks','transfusion','yeast','tic-tac-toe','diagnosis','facebook','horse-colic','anneal']
    
    #dataSets = ['hepati']#,'iris', 'pima-indians-diabetes']
    test_count = 1
    
    #for fileName in dataSets:
    #    print("Running for dataset: ", fileName)
    #100estimators #heart_tobetransformed_converted
    fileName='iris'
    dataset_name=fileName
    print("Running for dataset: ", fileName)
    acc_list=[]
    num_runs=1
    for num_run in range(1,num_runs+1):
        print("How many time")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~How many time",num_run)
                
        #input_name  =  'datasets_new_originalfiles/' + dataset_name + '.txt' #train file
        '''
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
        
        dforiginal=df
        dfnew=df
        dfnew = dfnew.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('', np.nan)
        
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
        
        
        X=pd.DataFrame(X)
        
        mask = X.applymap(lambda x: x is None)
        cols = X.columns[(mask).any()]
        for col in X[cols]:
            X.loc[mask[col], col] = ''              
        
        #Y = df.iloc[:,-1].values # Here first : means fetch all rows :-1 means except last column
        #X = df.iloc[:,:-1].values # : is fetch all rows 3 means 3rd column  
        
        #print("len(X)",len(X))
        #print("len(Y)",len(Y))
        '''
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
        y_train=dfnew.groupby(['label'] * dfnew.shape[1], 1).agg('last')
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
        print("len(X)",len(X))
        print("len(y_train)",len(y_train))
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
        print("len(X)",len(X))
        print("len(y_test)",len(y_test))
        X_test=pd.DataFrame(X)
        
        mask = X_test.applymap(lambda x: x is None)
        cols = X_test.columns[(mask).any()]
        for col in X_test[cols]:
            X_test.loc[mask[col], col] = ''         
        
        #END FOR MAKING TEST DATASET        
        
        
        #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle='true')
        #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=1,shuffle='true') 
        X_val=X_train
        y_val=y_train
        
        
        #print("len(X_train): ",len(X_train))
        #print("len(X_test): ",len(X_test))
        #print("len(y_train): ",len(y_train))
        #print("len(y_test): ",len(y_test))
        X_test_original=X_test
        df_Xtest=X_test_original
        test_name  =  'datasets_bagging/' +  dataset_name + '_testoriginal'+ str(num_run) +'.txt'
        df_Xtest.to_csv(test_name,sep=' ',index=False,header=False)  
        
        df3_test=y_test 
        test_name  =  'datasets_bagging/' +  dataset_name + '_ytestoriginal'+ str(num_run) +'.txt'
        df3_test.to_csv(test_name,sep=' ',index=False,header=False)          
                
        df_Xtrain=X_train                     
        test_name  =   'datasets_bagging/' + dataset_name + '_trainoriginal'+ str(num_run) +'.txt'
        df_Xtrain.to_csv(test_name,sep=' ',index=False,header=False)
        
        df3_test=y_train                     
        test_name  = 'datasets_bagging/' + dataset_name + '_ytrainoriginal'+ str(num_run) +'.txt'
        df3_test.to_csv(test_name,sep=' ',index=False,header=False)        
        
        '''
        
        df3_test=pd.DataFrame(list(X_test_original))              
        test_name  = 'datasets_new_boosting/' + dataset_name + '_testoriginal'+ str(num_run) +'.txt'
        df3_test.to_csv(test_name,sep=' ',index=False,header=False) 
        
        
        df3_test=pd.DataFrame(list(X_train))              
        test_name  = 'datasets_new_boosting/' + dataset_name + '_trainoriginal'+ str(num_run) +'.txt'
        df3_test.to_csv(test_name,sep=' ',index=False,header=False)         
        '''
        
        df1=pd.DataFrame(list(X_train))
        df2=pd.DataFrame(list(y_train))        
        Xtotal_subsample = pd.concat([df1,df2], axis=1)
        
        counter=1
        
        #MAKING VALIDATION SET
        
        df1_val=X_val
        df2_val=y_val            
        
        Xval_total = pd.concat([df1_val,df2_val], axis=1)
        
        
        Xval_total=Xval_total.values.tolist()
        #print("Xval_total: ")
        #print(Xval_total)
        val_name  = 'datasets_new_boosting/' + dataset_name + '_validation'+ str(num_run) +'.txt'
        for var in range(len(Xval_total)):
            
            Xval_total[var] = [x for x in Xval_total[var] if str(x) != 'nan' and str(x) != '']
        
        
        
        import csv
        
        with open(val_name, "w",newline='\n') as f:
            writer = csv.writer(f,delimiter=' ')
            writer.writerows(Xval_total)                        
        
        
        
        
        y_bagging=[]
        y_boosting=[]
        weight_dict={}
        weight_dict_original={}            
        
        
        
        num_estimators=26 #set this value
        
        X_train_updated = X_train
        X_test_updated = X_test
        y_train_updated = y_train
        y_test_updated = y_test
        y_boosting=[]
            
        for counter in range(1,num_estimators):
            #print("1??????X_train_updated :",X_train_updated)
            #print("1??????X_train:",X_train)            
            #weight_dict,X_train_updated, X_test_updated, y_train_updated, y_test_updated = update_dataset(weight_dict,X_train, X_test, y_train, y_test)
            #print("2??????X_train_updated: ",X_train_updated)            
            #print("2??????X_train",X_train)
            #print(weigh)
            #check this out kaunsa dataset aega
            
            #for val set
            df1=X_val
            df2=y_val            
            #df1=pd.DatFrame(list(X_train_updated))
            #df2=pd.DataFrame(list(y_train_updated))        
            Xtotal_subsample = pd.concat([df1,df2], axis=1)            
            Xtotal_subsample_list=Xtotal_subsample.values.tolist()
            
            random.shuffle(Xtotal_subsample_list)
            sample=subsample(Xtotal_subsample_list,1.00 ) #0.80
                       
            #Xtotal=pd.DataFrame(list(sample))
            Xtotal=sample 
            for var in range(len(Xtotal)):
                Xtotal[var] = [x for x in Xtotal[var] if str(x) != 'nan']
            
            train_name  =   'datasets_bagging/' + dataset_name + '_validation'+ str(counter) +'.txt'
            print("train_name: ",train_name)
            #print("Xtotal:")
            #print(Xtotal)
            Xtotal = [list(filter(None, lst)) for lst in Xtotal]
            #print("Xtotal:2")
            #print(Xtotal)            
            
            import csv
            
            with open(train_name, "w",newline='\n') as f:
                writer = csv.writer(f,delimiter=' ')
                writer.writerows(Xtotal)            
            
            
            
            #for train set
            df1=X_train
            df2=y_train            
            #df1=pd.DatFrame(list(X_train_updated))
            #df2=pd.DataFrame(list(y_train_updated))        
            Xtotal_subsample = pd.concat([df1,df2], axis=1)            
            Xtotal_subsample_list=Xtotal_subsample.values.tolist()
            
            random.shuffle(Xtotal_subsample_list)
            sample=subsample(Xtotal_subsample_list, 1.0)#0.80
                       
            #Xtotal=pd.DataFrame(list(sample))
            Xtotal=sample 
            for var in range(len(Xtotal)):
                Xtotal[var] = [x for x in Xtotal[var] if str(x) != 'nan']
            
            train_name  =   'datasets_bagging/' + dataset_name + '_train'+ str(counter) +'.txt'
            print("train_name: ",train_name)
            #print("Xtotal:")
            #print(Xtotal)
            Xtotal = [list(filter(None, lst)) for lst in Xtotal]
            #print("Xtotal:2")
            #print(Xtotal)            
            
            import csv
            
            with open(train_name, "w",newline='\n') as f:
                writer = csv.writer(f,delimiter=' ')
                writer.writerows(Xtotal) 
                
            #df.to_csv(train_name, sep='\t', encoding='utf-8')
            #Xtotal.to_csv(train_name,sep=' ',index=False,header=False)
            #creating test set: 
            
            df1_test=X_test
            df2_test=y_test            
            #df1_test=pd.DataFrame(list(X_test_updated))
            #df2_test=pd.DataFrame(list(y_test_updated))   
            Xtest_total = pd.concat([df1_test,df2_test], axis=1)
            Xtest_total=Xtest_total.values.tolist()
                
            Xtest_total = [list(filter(None, lst)) for lst in Xtest_total]
            test_name  =  'datasets_bagging/' +  dataset_name + '_test'+ str(counter) +'.txt'
            for var in range(len(Xtest_total)):
                
                Xtest_total[var] = [x for x in Xtest_total[var] if str(x) != 'nan']
            
            import csv
            
            with open(test_name, "w",newline='\n') as f:
                writer = csv.writer(f,delimiter=' ')
                writer.writerows(Xtest_total)
            #df.to_csv(train_name, sep='\t', encoding='utf-8')
            #Xtest_total.to_csv(test_name,sep=' ',index=False,header=False)    
              
            predictions,max_acc,hrs_acc=associative_classifier_test.main(dataset_name,counter)
            
            #weight_dict=boosting_func(predictions,X_train_updated, X_test_updated, y_train_updated, y_test_updated, sample_weight,weight_dict,X_test_original)
                        
            y_boosting.append(predictions)
            
            #y_bagging.append(predictions)
        
        print("^^^^^^^^^^^^^^^^^^^^^y_boosting: ",y_boosting)
        print("DONE")
        acc=bagging_func(y_boosting,len(X_test),y_test)   
        acc_list.append(acc)
    print("num_estimators: ",dataset_name,num_estimators)
    print("list of 10 accuracies: ",acc_list)
    print("finall accuracy for total runs is: ",(sum(acc_list)/num_runs))         
        #bagging_func(y_bagging,len(X_test),y_test)    
        
        
        
        
        
        
            
            
            

        


