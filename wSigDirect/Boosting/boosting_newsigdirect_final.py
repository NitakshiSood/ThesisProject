#  code working for boosting but testing set remains the same. 
#addedd a feature of estimator value of 0 it shoud stop the algorithm copied from version 10. does no good struggling with lists and pandas df.
#this one has has new way of findinf x and y so better than 10th version , much better thans 10 and 11. 
#This atleast runs for hepati. but some problem in classifier weight need to correct the algo.
#15th version had a prob. i didnt multiply weights with respective count in boosting function, where we divide to calc the estimated error,instead i just add the weights and then added count and multiplied them whihc is wrong
#tryng to optimize version 12
#working boosting #FINAL CODE FOR NEW RIPPER BOOSTING
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
import math
import random
from random import randint
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
sum_dictvalues=0;
count1={};
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
        if len(set(ypred_bagging[i])) == 4 :#SET THIS VALUE: #no. of estimators
            print( "No mode found" )
            print("setting value: statistics.mode(ypred_bagging[i]): ",ypred_bagging[i][0])
            ypred_final.append(ypred_bagging[i][0])
        else:
            #print("statistics.mode(ypred_bagging[i]): ",statistics.mode(ypred_bagging[i]))
            ypred_final.append(statistics.mode(ypred_bagging[i]))
    #print("ypred_final")
    #print(ypred_final)
    #print("len(ypred_final): ",type(ypred_final))
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
    #print("#######temp:")
    #print(temp)
    return sample


def merge(list1, list2): 
      
    merged_list = list(zip(list1, list2))
    return merged_list


def update_dataset(weight_dict,X_train, X_test, y_train, y_test, X_val, y_val):
    print("update_dataset")
     
    #old version with list
    test_name  = 'datasets_new_boosting/' + dataset_name + '_testoriginal'+ str(num_run) +'.txt' 
    with open(test_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dxy_test = [line.strip().split(sep) for line in data]
    
    train_name  = 'datasets_new_boosting/' + dataset_name + '_trainoriginal'+ str(num_run) +'.txt' 
    with open(train_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dxy_train = [line.strip().split(sep) for line in data]
    
    val_name  = 'datasets_new_boosting/' + dataset_name + '_valoriginal'+ str(num_run) +'.txt' 
    with open(val_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dxy_xval = [line.strip().split(sep) for line in data]     
    
    print(type(dxy_test))
    print(type(dxy_train))
    
    
    test_name  = 'datasets_new_boosting/' + dataset_name + '_ytestoriginal'+ str(num_run) +'.txt'
    with open(test_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dy_test = [line.strip().split(sep) for line in data]
    
    train_name  = 'datasets_new_boosting/' + dataset_name + '_ytrainoriginal'+ str(num_run) +'.txt' 
    with open(train_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dy_train = [line.strip().split(sep) for line in data] 
    
    val_name  = 'datasets_new_boosting/' + dataset_name + '_yvaloriginal'+ str(num_run) +'.txt' 
    with open(val_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dy_val = [line.strip().split(sep) for line in data]    
    
    y_test=pd.DataFrame(dy_test)
    y_train=pd.DataFrame(dy_train)
    y_val=pd.DataFrame(dy_val)
    
    #print("NOW FOCUSSSSSSS---------: ")
    for i in range(len(dxy_test)):
        temp=[]
        #print("ORIGINAL dxy_test[i]: ",dxy_test[i])
        for values in (dxy_test[i]):             
            #print("1values: ",values)
            #print("weight_dict[values]: ",weight_dict[values])
            values=(float(values)*(weight_dict[values]))
            if values== math.inf:
                print("WAS THIS REQUIRED?")
                values=1.7976931348623157e+300            
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
            if values== math.inf:
                print("WAS THIS REQUIRED?")
                values=1.7976931348623157e+300
            temp.append(values)
        dxy_train[i]=temp  
        #print("AFTER dxy_train[i]: ",dxy_train[i])        
     
     
    X_train=pd.DataFrame(dxy_train)
    
    #valset
    #print("1FOCUS HERE X_train: ",X_train)
    for i in range(len(dxy_xval)):
        temp=[]
        #print("ORIGINAL dxy_xval[i]: ",dxy_xval[i])
        for values in (dxy_xval[i]): 
         #   print("values: ",values)
            values=(float(values)*(weight_dict[values]))
            if values== math.inf:
                print("WAS THIS REQUIRED?")
                values=1.7976931348623157e+300
            temp.append(values)
        dxy_xval[i]=temp  
        #print("AFTER dxy_xval[i]: ",dxy_xval[i])        
     
     
    X_val=pd.DataFrame(dxy_xval)    
    
    
    
    return weight_dict,X_train, X_test, y_train, y_test, X_val, y_val

     

def boosting_func(predictions,X_train, X_test_updated, y_train, y_test, sample_weight,weight_dict,X_test):
    print("Begin Boosting: ")
    
    test_name  = 'datasets_new_boosting/' + dataset_name + '_ytestoriginal'+ str(num_run) +'.txt'
    with open(test_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dy_test = [line.strip().split(sep) for line in data]
    
    
    #y_test=dy_test
    y_test=pd.DataFrame(dy_test)
    #y_train=pd.DataFrame(dy_train)
    
    predictions=predictions.to_frame()
    print("predictions")
    #print(predictions)
    print("y_test")
    #print(y_test)    
    incorrect = predictions != y_test
    print("incorrect")
    #print(incorrect)
    
    #print("here X_test_updated: ",X_test_updated)
    estimator_error=0
    test_name  = 'datasets_new_boosting/' + dataset_name + '_testoriginal'+ str(num_run) +'.txt' 
    with open(test_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dxy = [line.strip().split(sep) for line in data]
    
        
    
    for i in range(incorrect.shape[0]):
        #print("?dxy[i]: ",dxy[i])
                
        Xtest_feature=list(set(dxy[i]))
        #print((incorrect))
        #print(float(incorrect[0][30]))
        #print(float(incorrect[0][10]))
        
        #print("?Xtest_feature: ",Xtest_feature)
        for feature in Xtest_feature:
            #print("feature: ",feature)
            #print("weight_dict[feature]: ",weight_dict[feature])
            #print("incorrect[i]: ",incorrect[0][i])
            if incorrect[0][i]==True: #missclassified
                #print("MISSCLASSIFIED")
                #print("weight_dict[feature]: ",weight_dict[feature])
                #print("feature: ",feature)
                #estimator_error=estimator_error +  0.5 - (1/2*(float(1)* weight_dict[feature]))
                estimator_error = estimator_error+ float(1)*weight_dict[feature]
                #print("estimator_error: ",estimator_error)
            else:
                #print("CORRECTCLASSIFIED")
                #print("weight_dict[feature]: ",weight_dict[feature])
                #print("feature: ",feature)
                #estimator_error=estimator_error + 0.5 - (1/2*(float(0)* weight_dict[feature]))
                estimator_error = estimator_error+ float(0)*weight_dict[feature]
                #print("estimator_error: ",estimator_error)
        
     
    #print("estimator_error:")  
    #print(estimator_error)
    # this hsould not be smaple weight, but the weight of all the train samples so change it
    sum_val=sum(weight_dict.values())
    print("weight_dict: ")
    print(weight_dict)
    print("sum_val:",sum_val)    
    print("estimator_error before division: ",estimator_error)
    
    
    sum_weight=0
    for value in weight_dict:
        if value!='':
            sum_weight=sum_weight+weight_dict[value]*count1[value]
    
    print("sum_weight: ",sum_weight)      
    
    
    print("incorrect.shape[0]: ",incorrect.shape[0])
    print()
    #print(weigh)
    #misclassification_rate = estimator_error / (sum_val*incorrect.shape[0])
    #misclassification_rate = estimator_error / (sum_val*sum_dictvalues)
    misclassification_rate = estimator_error / (sum_weight)
    print("misclassification_rate: ",misclassification_rate)
    estimator_error=misclassification_rate
    #print(weigh)
    
    #set this value
    n_classes_=10 #SET THIS VALUES #FOR IRIS ONLY #set this
    # if worse than random guess, stop boosting
    
    if estimator_error == 0: #i dont think that negative error woud be aprob
        # you need to change this
        print("HERE1???")
        return weight_dict, 10
    print("estimator_error: ",estimator_error)
    
    #not required. this is indeed the assumption by the algorithm
    #if estimator_error >= 1 - 1 / n_classes_:
    #    print("!!!!!!!!!!!!!!!!!!!!!goes here22222222222?")
    #    return None,None
    
    if (((1-estimator_error)/estimator_error) < 0):
        print("HERE2???")
        return weight_dict, None
    print("CALCULATING ALPHA VALUES?????????????")
    import math
    alpha=(1/2) *((np.log((1-estimator_error)/estimator_error)))
    #alpha = 0.5 * math.log((1.0 - estimator_error) / (estimator_error + 1e-10))
    print(alpha)    
    print("alpha: ",alpha)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@BEFORE THE UPDATE: ")
    print(weight_dict)
    for i in range(len(incorrect)):
        
        if incorrect[0][i]==True: #missclassified
            #sample_weight=np.exp(-(alpha)* (-1))
            Xtest_feature=list(set(dxy[i]))
            for feature in Xtest_feature:
                print("MISSCLASSIFIED")
                print("before 1 weight_dict[feature]: ",weight_dict[feature])
                #weight_dict[feature]=(weight_dict[feature]*np.exp(-(alpha)* (-1)))/sum_val
                weight_dict[feature]=(weight_dict[feature]*np.exp(-(alpha)* (-1)))/sum_val
                print("feature: ",feature)
                print("after 1 weight_dict[feature]: ",weight_dict[feature])
            #weight_dict = dict.fromkeys(feature_values, sample_weight)
        else:
            #sample_weight=sample_weight*np.exp(-(alpha)* (1))
            Xtest_feature=list(set(dxy[i]))
            for feature in Xtest_feature:
                print("CORRECTCLASSIFIED")
                print("before 2 weight_dict[feature]: ",weight_dict[feature])
                #weight_dict[feature]=(weight_dict[feature]*np.exp(-(alpha)* (1)))/sum_val
                weight_dict[feature]=(weight_dict[feature]*np.exp(-(alpha)* (1)))/sum_val
                print("feature: ",feature)
                print("after 2 weight_dict[feature]: ",weight_dict[feature])                
            
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@AFTER UPDATE, weights are boosting: ")
    print(weight_dict)
    #might have to change this thing
    for key, value in weight_dict.items():
        if value==0.0:
            weight_dict[key]= 0.1e-9  #0.1
    import math        
    for key, value in weight_dict.items():
        if value== math.inf:
            print("HERE???DID THE INF GET RECOG??")
            weight_dict[key]= 1.7976931348623157e+100  #np.nan_to_num(np.inf) #1.7976931348623157e+308
            
    #print(weigh)
    #replaced the zero problem
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@AFTER zero problem UPDATE, weights are boosting: ")
    print(weight_dict)    
    return weight_dict, alpha

def unique(list1): 
    x = np.array(list1) 
    print(np.unique(x)) 

def boost_pred(y_bagging, len_pred, Ytest,classifier_weight,counter):
    print("Hi")
    ypred_bagging=[]
    print("classifier_weight: ",classifier_weight)
    test_name  = 'datasets_new_boosting/' + dataset_name + '_ytestoriginal'+ str(num_run) +'.txt'
    with open(test_name, 'r') as f:
        data = f.read().strip().split('\n') 
    dy_test = [line.strip().split(sep) for line in data]
    
    Ytest=pd.DataFrame(dy_test)    
    ypred_bagging_withoutweight=[]
    for j in range(0,len_pred):
        temp=[]
        temp_2=[]
        
        
        #print(weigh)
        for i in range(len(y_bagging)):            
            cw=classifier_weight[i+1]
            val=float(y_bagging[i][j])*cw
            temp.append(val)
            temp_2.append(y_bagging[i][j])
        ypred_bagging.append(temp)
        ypred_bagging_withoutweight.append(temp_2)
    print("!!!!!!!!!! ypred_bagging !!!!!!!!!!!!") 
    print(ypred_bagging)
    print("!!!!!!!!!! ypred_bagging_withoutweight !!!!!!!!!!!!") 
    print(ypred_bagging_withoutweight)    
    ypred_final=[]
    for i in range(len(ypred_bagging)):
        #print("len_pred: 3")
        #print(len(set(ypred_bagging[i]))) 
        
        index=ypred_bagging[i].index(max(ypred_bagging[i]))
        ypred_final.append(ypred_bagging_withoutweight[i][index])
        
    #print("ypred_final")
    #print(ypred_final)
    
    #print("len(ypred_final): ",type(ypred_final))
    #print("Ytest")
    #print(Ytest)    
    #print("len(Ytest): ",type(Ytest))
    print()
    
    count=0
    for i in range(len(ypred_final)):
        #print("ypred_final[i]: ",ypred_final[i])
        #print("Ytest[0][i]: ",Ytest[0][i])
        if(ypred_final[i]==Ytest[0][i]):
            count=count+1
    print("count: ",count)
    print("bagging accuracy: ", (count/len(ypred_final)))
    #print(weigh)
    return ypred_final   

def f(x):
    if x.last_valid_index() is None:
        return np.nan
    else:
        return x[x.last_valid_index()]
    
def Merge(dict1, dict2): 
    return(dict2.update(dict1)) 

if __name__ == "__main__":
    
    # dataSets=['iris', 'monks','transfusion','yeast','tic-tac-toe','diagnosis','facebook','horse-colic','anneal']
    #dataSets = ['pima']#,'iris', 'pima-indians-diabetes']
    test_count = 1
    #print(wei)
    #https://scikit-learn.org/stable/modules/preprocessing.html
    
    #for fileName in dataSets:
    #    print("Running for dataset: ", fileName)
    #50estimatorsboosting
    fileName='led7'#heart_tobetransformed_converted
    dataset_name=fileName
    print("Running for dataset: ", fileName)
    
    num_runs=1
    for num_run in range(num_runs):
        print("How many time")
        #input_name  = 'datasets_new_originalfiles/' + dataset_name + '_tr'+ str(num_runs) +'.txt' #train file
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
        
        print("len(X)",len(X))
        print("len(Y)",len(Y))
        
        
        X=pd.DataFrame(X)
        
        mask = X.applymap(lambda x: x is None)
        cols = X.columns[(mask).any()]
        for col in X[cols]:
            X.loc[mask[col], col] = '' 
       
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle='true')
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=1,shuffle='true') 
        
        
        
        
        
        #print(X_train)
        
        print("len(X_train): ",type(X_train))
        print("len(X_test): ",len(X_test))
        print("len(y_train): ",len(y_train))
        print("len(y_test): ",len(y_test))
        X_test_original=X_test
        #df3_test=pd.DataFrame(list(X_test_original)) 
        
        df_Xtest=X_test_original
        test_name  = 'datasets_new_boosting/' + dataset_name + '_testoriginal'+ str(num_run) +'.txt'
        df_Xtest.to_csv(test_name,sep=' ',index=False,header=False)  
        
        df3_test=y_test 
        test_name  = 'datasets_new_boosting/' + dataset_name + '_ytestoriginal'+ str(num_run) +'.txt'
        df3_test.to_csv(test_name,sep=' ',index=False,header=False)          
        
        
        df_Xtrain=X_train                     
        test_name  = 'datasets_new_boosting/' + dataset_name + '_trainoriginal'+ str(num_run) +'.txt'
        df_Xtrain.to_csv(test_name,sep=' ',index=False,header=False)
       
        
        df_Xval=X_val
        val_name  = 'datasets_new_boosting/' + dataset_name + '_valoriginal'+ str(num_run) +'.txt'
        print("val_name: ",val_name)
        df_Xval.to_csv(val_name,sep=' ',index=False,header=False)
        
        df_Yval=y_val
        val_name  = 'datasets_new_boosting/' + dataset_name + '_yvaloriginal'+ str(num_run) +'.txt'
        print("val_name: ",val_name)
        
        df_Yval.to_csv(val_name,sep=' ',index=False,header=False)        
        
        
        df3_test=y_train                     
        test_name  = 'datasets_new_boosting/' + dataset_name + '_ytrainoriginal'+ str(num_run) +'.txt'
        df3_test.to_csv(test_name,sep=' ',index=False,header=False)        
        
        
        
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
        val_name  = 'datasets_new_boosting/' + dataset_name + '_validation'+ str(counter) +'.txt'
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
        
        frames = pd.concat([df_Xtest,df_Xtrain,df_Xval], axis=0)
        #frames=dforiginal
        
        mask = frames.applymap(lambda x: x is None)
        cols = frames.columns[(mask).any()]
        for col in df[cols]:
            frames.loc[mask[col], col] = ''
        frames_list= list(frames.values.flatten()) 
        feature_values=list(filter(None, frames_list))
        n=len(feature_values)
        print()
        print("feature_values",len(feature_values))
        print(feature_values)
        print("df_Xtrain")
        print(df_Xtrain)
        print(df_Xtrain.count())
        print(type(df_Xtrain.count()))
        
        count_unique=len(set(feature_values))
        print("count_unique: ",count_unique)
        #sample_weight=1/(X_train.shape[0])
        sample_weight=1/(count_unique)
        
        
        count2={}
        #********************calc totl xtrain
        print("X_test: ")
        #print(X_test.shape)
        #print(X_test.shape[1])
        print(X_test)
        print(type(X_train))
        from collections import Counter
        for i in range(0,X_test.shape[1]):
            
            
            if i ==0:
                #A=X_test[i].value_counts().to_dict() 
                count1 = Counter(X_test[i].value_counts().to_dict())
                #count1 = X_test[i].value_counts().to_dict()
                print("count1: first: ",count1)
            else:
                #count2 = X_test[i].value_counts().to_dict()
                count2 = Counter(X_test[i].value_counts().to_dict())
                count1=count2+count1
                #Merge(count2,count1)
                #print("count1: second: ",count1)
                
    
        print("count1")
        print(count1) #still not correct completely.
        
        
        
        
        #print(X_train)
        #print("X_train.shape: ",X_train.shape)
        #print("X_train.shape[0]: ",X_train.shape[0])
        
        #sample_weight=1
        print("sample_weight: ",sample_weight)
       
        weight_dict = dict.fromkeys(feature_values, (round(sample_weight,2)))
        weightcount_dict = dict.fromkeys(feature_values, 0)
        print("len(weight_dict.keys()): ",len(weight_dict.keys()))
        print("weight_dict: ")
        print(weight_dict)
        
        #for calculating the denominator
        #global sum_dictvalues
        print("number of spaces herre ", count1[''])
        sum_weight=0
        for value in weight_dict:
            if value!='':
                sum_weight=sum_weight+weight_dict[value]*count1[value]
        
        print("sum_weight: ",sum_weight)   
        sum_dictvalues=sum_weight
        #sum_dictvalues = sum(count1.values())-count1['']
        print("sum_dictvalues: ",sum_dictvalues)
                
        #print(weigh)
        num_estimators=5 #set this
        #print(weigh)
        X_train_updated = X_train
        X_test_updated = X_test
        y_train_updated = y_train
        y_test_updated = y_test
        y_boosting=[]
        classifier_weight={} 
        classi=[]
        for i in range(1,num_estimators):
            classi.append(i)
        classifier_weight = dict.fromkeys(classi, 0)
        print("$$$$$$$$classifier_weight: ",classifier_weight)
        alpha=0
        for counter in range(1,num_estimators):
            
            #print("1??????X_train_updated :",X_train_updated)
            #print("1??????X_train:",X_train)
            #print("1??????y_train:",y_train)
            print("CHECK CHECK @@@@@@@ before updateweight_dict:, ",weight_dict)
            weight_dict,X_train_updated, X_test_updated, y_train_updated, y_test_updated, X_val_updated, y_val_updated = update_dataset(weight_dict,X_train, X_test, y_train, y_test, X_val, y_val)
            #print("2??????X_train_updated: ",X_train_updated)            
            #print("2??????X_train",X_train)
            #print("1??????y_train:",y_train_updated)
            
            
            #validation set
            
            df1=X_val_updated
            df2=y_val_updated  
            Xtotal_subsample = pd.concat([df1,df2], axis=1)
            Xtotal_subsample_list=Xtotal_subsample.values.tolist()
            
            random.shuffle(Xtotal_subsample_list)
            sample=subsample(Xtotal_subsample_list, 0.80)
            print(type(sample))           
            
            Xtotal=sample
            #print("Xtotal: ")
            #print(Xtotal)
            #print(type(Xtotal))
            for var in range(len(Xtotal)):
                Xtotal[var] = [x for x in Xtotal[var] if str(x) != 'nan']
            
            train_name ='datasets_new_boosting/' + dataset_name + '_validation'+ '1' +'.txt'
            print("train_name: ",train_name)
            
            import csv
            
            with open(train_name, "w",newline='\n') as f:
                writer = csv.writer(f,delimiter=' ')
                writer.writerows(Xtotal)                        
            
            
            #trainset
            
            #check this out kaunsa dataset aega
            #df1=pd.DataFrame(list(X_train_updated))
            #df2=pd.DataFrame(list(y_train_updated))  
            df1=X_train_updated
            df2=y_train_updated  
            Xtotal_subsample = pd.concat([df1,df2], axis=1)
            Xtotal_subsample_list=Xtotal_subsample.values.tolist()
            
            random.shuffle(Xtotal_subsample_list)
            sample=subsample(Xtotal_subsample_list, 1.0) #0.80
            print(type(sample))           
            
            Xtotal=sample
            #print("Xtotal: ")
            #print(Xtotal)
            #print(type(Xtotal))
            for var in range(len(Xtotal)):
                Xtotal[var] = [x for x in Xtotal[var] if str(x) != 'nan']
            
            train_name  = 'datasets_new_boosting/' + dataset_name + '_train'+ str(counter) +'.txt'
            print("train_name: ",train_name)
            
            import csv
            
            with open(train_name, "w",newline='\n') as f:
                writer = csv.writer(f,delimiter=' ')
                writer.writerows(Xtotal)            
           
            df1_test=X_test_updated
            df2_test=y_test_updated            
            
            Xtest_total = pd.concat([df1_test,df2_test], axis=1)
            
            
            #random.shuffle(Xtest_total)
            #sample=subsample(Xtest_total, 1.0)            
            Xtest_total=Xtest_total.values.tolist()
            
            test_name  = 'datasets_new_boosting/' + dataset_name + '_test'+ str(counter) +'.txt'
            #Xtest_total.to_csv(test_name,sep=' ',index=False,header=False) 
            #df.to_csv(train_name, sep='\t', encoding='utf-8')
            for var in range(len(Xtest_total)):
                
                Xtest_total[var] = [x for x in Xtest_total[var] if str(x) != 'nan']
            
            
            
            import csv
            
            with open(test_name, "w",newline='\n') as f:
                writer = csv.writer(f,delimiter=' ')
                writer.writerows(Xtest_total)                        
            
            
            
            #print(weigh)   
            #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!COUNTER: ",counter)
            predictions=associative_classifier_test.main(dataset_name,counter)
            #print("predictions: ")
            #print(predictions)
            #print(type(predictions))
            
            weight_dict_original= weight_dict
            alpha_original= alpha
            
            #weight_dict, alpha=boosting_func(predictions,X_train_updated, X_test_updated, y_train_updated, y_test_updated, sample_weight,weight_dict,X_test_original)
            weight_dict, alpha=boosting_func(predictions,X_train, X_test, y_train, y_test, sample_weight,weight_dict,X_test_original)
            print()
            if weight_dict== None or alpha==None:
                weight_dict=weight_dict_original
                alpha=0
                #alpha=alpha_original
                #classifier_weight[counter]= alpha           
                #y_boosting.append(predictions)                
                #break
            classifier_weight[counter]= alpha           
            y_boosting.append(predictions)
            
            #y_bagging.append(predictions)
        
        #print("^^^^^^^^^^^^^^^^^^^^^y_boosting: ",y_boosting)
        print("classifier_weight: ",classifier_weight)
        print("y_boosting DONE, now prediction!!")
        boost_pred(y_boosting,len(X_test),y_test,classifier_weight,counter)                

        
        
        
        
        
        
            
            
            

        


