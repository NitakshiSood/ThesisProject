import sklearn
from sklearn import preprocessing
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
import numpy as np
#now aim is to add the newly added colms only to this. so that we can continue the naming scheme
# this code works. but it wont work when you call data tranformationm after splitting into pieces. so this is not a good idea.
import csv
#nameFile="D:/Thesis/summer_2019/sigdirect_float_Try - Copy\datasets_new_originalfiles\WCB.txt"

def transformdata(X1,original_feature_length,df_y_train):
    #print("original_feature_length: ",original_feature_length)
    x_new1=X1.iloc[:,original_feature_length:] #34
    x_new1=x_new1.astype(int)
    #f=df.iloc[:,1:]
    enc = sklearn.preprocessing.OrdinalEncoder()
    enc.fit(x_new1)
    x_new=enc.transform(x_new1)    
    x_new=pd.DataFrame(x_new)
    
    temp=list(range(0,X1.shape[1]))
    #df = pd.DataFrame(np.random.randn(8, 4),columns=temp)  
    X1.columns=temp
    X_transformed_perrange=x_new
    count=X1[original_feature_length-1].max() #33
    #print("x_new: ",x_new)
    #print("count: ",count)
    #print("x_new.shape[1] :",x_new.shape[1])
    for i in range(x_new.shape[1]):
        #print("x_new[i]: ",x_new[i])
        #print("X_transformed_perrange[i]: ",X_transformed_perrange[i])
        #print("count: ",count)
        #print("x_new[i]+count +1: ",x_new[i]+156)
        counter=int(count)+1
        #print("x_new[i]+count +1: ",x_new[i]+int(counter))
        X_transformed_perrange[i]=x_new[i]+ int(counter)#156
            #count=X_transformed[i].nunique()+
        count=X_transformed_perrange[i].max()
        #print(weigh)
    
    X_transformed_perrange=X_transformed_perrange.astype(int)
    #print("X_transformed_perrange: ",X_transformed_perrange)  
    #print("X1[:,0:33]: ",X1.iloc[:,0:original_feature_length])
    #X_transformed_perrange=pd.concat([X1.iloc[:,0:33], X_transformed_perrange], axis=1, sort=False) 
    X_transformed_perrange=pd.concat([X1.iloc[:,0:original_feature_length], X_transformed_perrange], axis=1, sort=False) 
    #print("last col index: ",i)    
    count=X_transformed_perrange.iloc[:,-1].max()
    counter=int(count)+1
    
    #for y train
    enc = sklearn.preprocessing.OrdinalEncoder()
    enc.fit(df_y_train)
    df_y_train=enc.transform(df_y_train)    
    df_y_train=pd.DataFrame(df_y_train)    
    
    #print("df_y_train: ",df_y_train)
    #print("type of df_y_train: ",type(df_y_train))
    #print("counter: ",counter)
    #print("type of counter: ",type(counter))
    df_y_train=df_y_train+int(counter) #['label']
    #print("final X_transformed_perrange: ",X_transformed_perrange) 
    #print("df_y_train: ",df_y_train)
    #print(checkcheck)
    temp=list(range(0,X_transformed_perrange.shape[1]))
    #df = pd.DataFrame(np.random.randn(8, 4),columns=temp)  
    X_transformed_perrange.columns=temp    
    return X_transformed_perrange,df_y_train
import numpy as np

if __name__ == '__main__':
    ''' Note:
    to understand and separate different item types (original, and my ordered version)
    original should be string since it is read from file, and it must be categorical at the 
    end of the day. And thus mine can be integer.\n1 \n1 \n1 \n1 \n1 \n1 \n1 \n1 \n1 \n1 
    '''

def main(X1,original_feature_length,df_y_train):
    #X_transformed_perrange=transformdata(X)
    print("********************ORIGINAL : ",X1)
    print("********************ORIGINAL--------df_y_train: ",df_y_train)
    X_transformed_perrange,df_y_train=transformdata(X1,original_feature_length,df_y_train)
    print("********************AFTERRRRRRRR : ",X_transformed_perrange)
    print("********************AFTERRRRRRRR-----------df_y_train: ",df_y_train)    
    return X_transformed_perrange,df_y_train






'''


#nameFile='datasets_new_originalfiles/' + 'WATT_Data_March14_2018_green_no_assessment_positive_train_lessmissing_boruta_durationDisc_numenc_mean_modified'+'.csv' #csv
fileContent=open(nameFile,"r")
fileCont=fileContent.readlines()
dataset=[]

for row in fileCont:
    row=row.strip()
    #row=row.replace("?"," ")
    dataset.append(row.split(''))

print("dataset")
print(dataset)
#method2
#X=dataset
enc = sklearn.preprocessing.OrdinalEncoder()
print("X BEFORE")
print(X)
enc.fit(X)
X_transformed=enc.transform(X)
print("X AFTER")
X_transformed=pd.DataFrame(X_transformed)
print(X_transformed)
#print(weigh)
test_name  = 'datasets_new_boosting/' + 'just testing2_file2' + '.txt'
X_transformed.to_csv(test_name,sep=' ',index=False,header=False)
count=1
X_transformed_perrange = X_transformed
test_name  = 'datasets_new_boosting/' + 'just testing2_file3' + '.txt'
print("X_transformed: ")
print(X_transformed)
temp=[]
for i in range(X_transformed.shape[1]):
    if i==0:
        X_transformed_perrange[i]=X_transformed[i]+count
        #count=X_transformed[i].nunique()
      
    else:
        X_transformed_perrange[i]=X_transformed[i]+count+1
        #count=X_transformed[i].nunique()+
    count=X_transformed_perrange[i].max()
   #print("count: ",count)
   #print(weigh)
   #print("X_transformed[i].nunique()",X_transformed[i].nunique())
   
    temp.append(count)
X_transformed_perrange=X_transformed_perrange.astype(int)
X_transformed_perrange.to_csv(test_name,sep=' ',index=False,header=False)
print("X_transformed_perrange")
print(X_transformed_perrange)
print("temp: ")
print(temp)
'''