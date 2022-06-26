# just for test
import random
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import miniproject as ml
from sklearn.utils import resample
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics
from random import choices
import miniproject as ml
import statistics
# import Logistic as NN
# from src.NeuralNetwork import NeuralNetwork
# import src.utils as utils
import neuralNW
import associative_classifier_test
import preprocessing
import warnings
warnings.filterwarnings('ignore')

#https://www.datacamp.com/community/tutorials/adaboost-classifier-python

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
    from efficient_apriori import apriori

    # dataSets=['iris', 'monks','transfusion','yeast','tic-tac-toe','diagnosis','facebook','horse-colic','anneal']
    dataSets = ['anneal']#,'iris', 'pima-indians-diabetes']
    for fileName in dataSets:
        print("Running for dataset: ", fileName)

        associative_classifier_test.main(fileName)

        filenm = 'datasets/' + fileName + '_train1.txt.rules'
        fileContent = open(filenm, "r")
        fileCont = fileContent.readlines()
        dataset = []
        dictionaryOfRules = dict()
        i=0
        for row in fileCont:
            row=row.strip()
            #row=row.replace("?"," ")
            rule=row.split(";")
            rule_rhs=rule[0].split(" ")[-1]
            rule_lhs=rule[0].split(" ")[:-1]            
            row=row[row.index(";")+1:]
            stats=row.split(",")
            support=stats[0]
            confidence=stats[1]
            pval=stats[2]
            i+=1
            dataset.append(row.split(' '))
            dictionaryOfRules[i] = [rule_lhs]
            dictionaryOfRules[i].append(float(confidence))
            dictionaryOfRules[i].append(float(support))
            dictionaryOfRules[i].append(float(pval))
            dictionaryOfRules[i].append(rule_rhs)



        # print(Xtrain)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print(Ytrain)
        print(len(dictionaryOfRules))
        print("Dictionary of rules")
        print(dictionaryOfRules)

        Xtrain=[]
        Ytrain=[]

        filenm = 'datasets/' + fileName + '_train2.txt'
        fileContent = open(filenm, "r")
        fileCont = fileContent.readlines()
        train_dataset = []
        for row in fileCont:
            row = row.strip()
            train_dataset.append(row.split(" "))

        for row in range(len(train_dataset)):
            currentRow = train_dataset[row]
            temp = []
            # for col in range(len(currentRow)-1):
            # col=0
            # print("Row number ",row,": ",currentRow)
            for i in range(1, len(dictionaryOfRules) + 1):
                # print("++++",dictionaryOfRules[i][0])
                # print("====",currentRow[0:4])
                present = 0
                # print("===",len(dictionaryOfRules[i][0]))
                temp.append(dictionaryOfRules[i][1])
                temp.append(dictionaryOfRules[i][2])
                temp.append(dictionaryOfRules[i][3])

                for tup in range(len(dictionaryOfRules[i][0])):
                    if (dictionaryOfRules[i][0][tup] in currentRow[:-1]):
                        present += 1
                if (present == len(dictionaryOfRules[i][0])):
                    # print("yes")
                    temp.append(1)
                else:
                    temp.append(0)

            Xtrain.append(temp)
            Ytrain.append(currentRow[-1][1])

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # print(Xtrain)
        # print("====")
        # print(Ytrain)

        Xtest = []
        Ytest = []
        filenm = 'datasets/' + fileName + '_test.txt'
        fileContent = open(filenm, "r")
        fileCont = fileContent.readlines()
        test_dataset = []
        for row in fileCont:
            row = row.strip()
            test_dataset.append(row.split(" "))

        for row in range(len(test_dataset)):
            currentRow = test_dataset[row]
            temp = []
            # for col in range(len(currentRow)-1):
            # col=0
            # print("Row number ",row,": ",currentRow)
            for i in range(1, len(dictionaryOfRules) + 1):
                # print("++++",dictionaryOfRules[i][0])
                # print("====",currentRow[0:4])
                present = 0
                # print("===",len(dictionaryOfRules[i][0]))
                temp.append(dictionaryOfRules[i][1])
                temp.append(dictionaryOfRules[i][2])
                temp.append(dictionaryOfRules[i][3])

                for tup in range(len(dictionaryOfRules[i][0])):
                    if (dictionaryOfRules[i][0][tup] in currentRow[0:4]):
                        present += 1
                if (present == len(dictionaryOfRules[i][0])):
                    # print("yes")
                    temp.append(1)
                else:
                    temp.append(0)

            Xtest.append(temp)
            Ytest.append(currentRow[-1][1])
        print(len(Xtrain))
        print("============")
        # print(Ytest)
        # neuralNW.NeuralNetwork(Xtrain,Xtest,Ytrain,Ytest)

        '''
        svm_accuracy_list=[]
        #Best hyper parameter returned from the cross validation
        svm_best_paras = svm_train_cv(Xtrain, Ytrain)

        # Using the best hyper parameter to train svm using the traing dataset, and return accuracy and f1 score on the testing dataset
        svm_acc = svm_stats(Xtrain, Ytrain, Xtest, Ytest, svm_best_paras)
        svm_accuracy_list.append(svm_acc)
        #svm_f1_list.append(svm_f1)
        '''
        # Xtrain, Ytrain, Xtest, Ytest = loadDataSet()

        # Combine the traing set and the testing set
        

        num_runs = 8
        
        #neuralNW.NeuralNetwork(Xtrain, Xtest, Ytrain, Ytest)
        
        
        #PCA trial
        from sklearn.decomposition import PCA
        
        pca = PCA(.95)
        pca.fit(Xtrain)
        Xtrain = pca.transform(Xtrain)
        Xtest = pca.transform(Xtest)
        
        # Combine the traing set and the testing set
        Xtotal = np.append(Xtrain, Xtest, axis=0)
        Ytotal = np.append(Ytrain, Ytest, axis=0)        
        
        svm_accuracy_list = []
        svm_f1_list = []

        # Split the total data (280000) into the training set (240000) and the testing set (40000)
        # 6 folds used as the training dataset, 1 fold used as the testing dataset
        skf = StratifiedKFold(n_splits=5, shuffle=True)

        for i in range(num_runs):
            # Reshuffle the total number of data (280000) into the training set (240000) and the testing set (40000)
            
            for train_index, test_index in skf.split(Xtotal, Ytotal):
                # 6 folds used as the training set
                print("New TEST FORMED")
                Xtest = Xtotal[test_index]
                Ytest = Ytotal[test_index]                
                
                Xtrain = Xtotal[train_index]
                Ytrain = Ytotal[train_index]
                print("FOCUS")
                
                bagging_trainset=merge(Xtrain, Ytrain)
                
                print("~~~~~~~~ bagging_trainset ~~~~~~~~~~~~~~~~~~~~``")
                print(type(bagging_trainset))
                print(bagging_trainset[0])
                y_bagging=[]
                for bagging_var in range(0,1):                    
                    print("bagging_var: ",bagging_var)
                    trainDataset= resample(bagging_trainset, replace=True)
                    print("~~~~~~~~~~~~~trainDataset~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    print(trainDataset[0])
                    print(type(trainDataset))
                    Xtrain=[item[:-1] for item in trainDataset]
                    Ytrain=[item[-1] for item in trainDataset]
                    
                                   
                    # The remaining fold used as the testing set
                    
                    print(type(Xtrain[0]))
                    print(type(Xtest[0]))
                    print(type(Xtrain))
                    print(type(Xtest))                
                    #print(type(Ytrain))
                    print("@@@@@@@@@@@@@@@@@@@@@")
                    
                    Xtrain=np.asarray(Xtrain[:])
                    Ytrain=np.asarray(Ytrain)
                    
                    print(type(Xtrain[0]))
                    print(type(Xtest[0]))
                    print(type(Xtrain))
                    print(type(Xtest))
                    
                    print(Xtrain[0:2])
                    print("******now test********")
                    print(Xtest[0:2])
                    print("len(Xtrain): ",len(Xtrain))
                    print("len(Xtest): ",len(Xtest))
                    #print(Ytest)
                    Xtrain = Xtrain.reshape((Xtrain.shape[0], -1))
                    print("AFTER RESHAPING")
                    print(Xtrain)
                    # Flatten each image
                    #Xtrain = Xtrain.reshape((Xtrain.shape[0], -1))
                    #Xtest = Xtest.reshape((Xtest.shape[0], -1))
    
                    # For SVM
                    # Best hyper parameter returned from the cross validation
                    svm_best_paras, y_pred_internal = ml.svm_train_cv(Xtrain, Ytrain)
                    print("~~~~~~~~~~~~~~~~~~~~- y_pred -~~~~~~~~~~~~~~~")
                    
                    # Using the best hyper parameter to train svm using the traing dataset, and return accuracy and f1 score on the testing dataset
                    svm_acc, svm_f1,y_predd = ml.svm_stats(Xtrain, Ytrain, Xtest, Ytest, svm_best_paras)
                    svc=SVC(probability=True, kernel='linear')
                    #abc =AdaBoostClassifier(n_estimators=160, base_estimator=svc,learning_rate=0.5,algorithm="SAMME") #initially rate was 1
                    abc =AdaBoostClassifier(n_estimators=160, learning_rate=0.5,algorithm="SAMME")
                    model = abc.fit(Xtrain, Ytrain)
                    y_pred = model.predict(Xtest)
                    print("%%%%%%%%%%%%%%%%%%Accuracy:",metrics.accuracy_score(Ytest, y_pred))
                    
                    #print(y_pred)
                    y_bagging.append(y_pred)                    
                    
                    svm_accuracy_list.append(svm_acc)
                    svm_f1_list.append(svm_f1)
    
                
                print("######NOW PRINTING y_bagging: #####")
                #print(y_bagging)
                #print(type(y_bagging))
                #print(type(y_bagging[0]))
                #print(y_bagging[0][0])
                #print(y_bagging[1][0])
                #print(y_bagging[2][0])
                #bagging_func(y_bagging,len(Xtest),Ytest)
                break            
            
            
            
            

        #neuralNW.NeuralNetwork(Xtrain, Xtest, Ytrain, Ytest)


