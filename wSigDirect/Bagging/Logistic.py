# just for test
import random
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
import miniproject as ml
# Import Support Vector Classifier
from sklearn.svm import SVC
from sklearn.utils import resample
import xgboost as xgb
from random import choices
import miniproject as ml
from sklearn import metrics
import statistics
from sklearn.model_selection import cross_val_score
# import Logistic as NN
# from src.NeuralNetwork import NeuralNetwork
# import src.utils as utils
from sklearn.ensemble import GradientBoostingClassifier as GBC
import neuralNW
from sklearn.metrics import confusion_matrix
import associative_classifier_test
import preprocessing
import warnings
warnings.filterwarnings('ignore')

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
    dataSets = ['iris']#,'iris', 'pima-indians-diabetes'] ['flare,'breast','glass','hepati']
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
        print("Number of rules: ",len(dictionaryOfRules))
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
        #print(Ytest)
        #print(Ytrain)
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
                #print(skf.split(Xtotal, Ytotal))
                #print("train_index: ")
                #print(train_index)
                #print("test_index: ")
                #print(test_index)
                print("New TEST FORMED")
                Xtest = Xtotal[test_index]
                Ytest = Ytotal[test_index]                
                
                Xtrain = Xtotal[train_index]
                Ytrain = Ytotal[train_index]
                print("FOCUS")
                #svc=SVC(probability=True, kernel='linear')
                #gbc=AdaBoostClassifier(base_estimator=svc,learning_rate = 0.05)
                #gbc=AdaBoostClassifier(learning_rate = 0.05)
                
                print(" Xtrain is: ", )
                print(Xtrain)
                print("Ytrain is: ", )
                print(Ytrain)
                print("Xtest is: ", )
                print(Xtest)
                print("Ytest is: ", )
                print(Ytest)                
                
                #XGBCLASSIFIER
                #https://stackoverflow.com/questions/33749735/how-to-install-xgboost-package-in-python-windows-platform 
                #https://www.datacamp.com/community/tutorials/xgboost-in-python 
                #https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
                #xg_reg = xgb.XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)                 
                xg_reg = xgb.XGBClassifier()
                xg_reg.fit(Xtrain,Ytrain)
                y_pred = xg_reg.predict(Xtest)
                print("this is y pred: ")
                print(y_pred)
                print("this is Ytest: ")
                print(Ytest)                
                xg_reg.fit(Xtrain,Ytrain)
                
                y_pred = xg_reg.predict(Xtest)                
                #gbc = GBC(learning_rate = 0.01) #so far the best 
                #gbc.fit(Xtrain,Ytrain)
                #y_pred = gbc.predict(Xtest)
                
                cm = confusion_matrix(Ytest,y_pred)
                print(cm)
                #print('Accuracy = ', end = "")
                #print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))                
                
                #print("Accuracy score (Testing): {0:.3f}".format(gbc.score(Xtest, Ytest)))
                
                # Model Accuracy, how often is the classifier correct?
                print("different way to calculate Accuracy:",metrics.accuracy_score(Ytest, y_pred))
                
                #accuracies = cross_val_score(estimator = gbc, X = X, y = y, cv = 10, n_jobs = -1)
                #print('Accuracy =', accuracies.mean()*100, '%')
                #print('Standard Deviation =', accuracies.std()*100, '%')                
                
                break


