import pandas as pd
import random
import numpy as np
import pandas as pd

dataSets=['iris']
for fileName in dataSets:
    # Location of dataset
   
    
    
    # Location of dataset
    #url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
    
    # Assign colum names to the dataset
    #names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    
    #hepati
    #names= ['AGE','SEX','STEROID','ANTIVIRALS','FATIGUE','MALAISE','ANOREXIA','LIVER BIG','LIVER FIRM','SPLEEN PALPABLE','SPIDERS','ASCITES','VARICES','BILIRUBIN','ALK PHOSPHATE','SGOT','ALBUMIN','PROTIME','HISTOLOGY','class']
            # Read dataset to pandas dataframe
    #irisdata = pd.read_csv(url, names=names)   
    
    #dataSets=['iris', 'monks','transfusion','yeast','tic-tac-toe','diagnosis','facebook','horse-colic','anneal']
    dataSets=['anneal']
    for fileName in dataSets:
        #dataset= [('me','1', 'high'), ('me', '1', 'high'), ('me', '2', 'high'), ('you','2', 'high'), ('you', '2', 'high'),('you', '2', 'low'), ('you', '3', 'low'), ('you', '3', 'low'), ('me', '1', 'low'), ('us', '2', 'low')]
        print("Running for dataset: ", fileName)
        textFile='D:\Thesis\summer_2019\WCB_testing_foronlysigdirect_changepruningtech_likeripper_v3_UCIdatasets\datasets_new_originalfiles'+'\\' + fileName + '.csv'
        #textFile="D:\winter2019\DataMining\project\learntolearn\ourcode\classbased\datasets" +"\\"+fileName+".data"
        #nameFile="D:\winter2019\DataMining\project\learntolearn\ourcode\classbased\datasets" +"\\"+fileName+".names"
        fileContent=open(textFile,"r")
        fileCont=fileContent.readlines()
        dataset=[]
        for row in fileCont:
            row=row.strip()
            #row=row.replace("?"," ")
            dataset.append(row.split(','))

        random.shuffle(dataset)

        print("type of dataset: ",type(dataset))
        length=len(dataset)//4
        print("PRINT dataset:",dataset[0])
        #print(weigh)
        print(dataset)
        print("length of dataset: ", len(dataset))
        '''
        colContent = open(nameFile, "r")
        colCont = colContent.readlines()
        columns=[]
        for row in colCont:
            row=row.strip()
            columns=row.split(',')
            break
        print(columns)
        print("Names")
        names=columns
        for row in dataset:
            #for col in range(len(row)):
            col=0
            colVal=0
            while col < len(row) and colVal<len(columns):
                if(row[col] == "?"):
                    del(row[col])
                    colVal+=1
                else:
                    #row[col]=(str(columns[colVal]),row[col])
                    row[col]=(str(row[col]))
                    col+=1
                    colVal+=1
                    '''
    print("NEW DATASET:")
    print(dataset)
    #print(weigh)        
    # Read dataset to pandas dataframe
    irisdata = pd.DataFrame.from_records(dataset, columns=names)
    # Assign data from first four columns to X variable
    
    print("@@@@@@irisdata@@@@@")
    print(irisdata)        
    
    
    #X = irisdata.iloc[:, 0:4]
    X = irisdata.iloc[:, :-1]
    print("X VALUES IS")
    print(X)
    
    # Assign data from first fifth columns to y variable
    #y = irisdata.select_dtypes(include=[object]) 
    
    
    y=irisdata.iloc[:,-1]
    
    print("Y VALUE IS")
    print(y)
    
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
        
    from sklearn import preprocessing  
    #le = preprocessing.LabelEncoder()
    #y = y.apply(le.fit_transform) 
    
    #print("THE NEW Y IS :")
    #print(y)
    
    from sklearn.preprocessing import StandardScaler  
    #scaler = StandardScaler()  
    #scaler.fit(X_train)
    
    le = preprocessing.LabelEncoder()
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
     
         
    #X_train = scaler.transform(X_train)  
    #X_test = scaler.transform(X_test)  
    from sklearn.neural_network import MLPClassifier  
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
    mlp.fit(X_train, y_train.values.ravel())  
    
    predictions = mlp.predict(X_test) 
    
    from sklearn.metrics import classification_report, confusion_matrix  
    print(confusion_matrix(y_test,predictions))  
    print(classification_report(y_test,predictions))
    print("Accuracy of testing using NN: ",mlp.score(X_test,y_test))