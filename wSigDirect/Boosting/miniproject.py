

"""**SVM**"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.classification import f1_score

def supportVecMachine(X_train, y_train, X_test, y_test, reg_para_val, kernel_val, gamma_val):
  svclassifier = SVC(C=reg_para_val, kernel=kernel_val, gamma=gamma_val, decision_function_shape='ovo')
  svclassifier.fit(X_train, y_train)
  
  y_pred = svclassifier.predict(X_test) 
  
  acc = accuracy_score(y_test,y_pred)
  f1_score_val = f1_score(y_test, y_pred, average='macro')
  
  return acc, f1_score_val, y_pred

def svm_train_cv(X_train, y_train):
  reg_parameter_list = [0.001, 0.01, 0.1, 1]
  kernel_list = ['rbf', 'linear', 'sigmoid']
  gamma_list = [0.0001, 0.001, 0.01, 0.1]
  
  # 10-fold internal stratified cross-validation
  skf = StratifiedKFold(n_splits=10, shuffle=True)
  
  max_acc = 0.0
  max_reg_para = []
  
  for reg_para in reg_parameter_list:
    for kernel_val in kernel_list:
      for gamma_val in gamma_list:
    
        total_acc = 0.0

        for train_index, test_index in skf.split(X_train, y_train):
          # 9 folds used as the training set
          valid_X_train = X_train[train_index]
          valid_y_train = y_train[train_index]
          # The remaining fold used as the testing set
          valid_X_test = X_train[test_index]
          valid_y_test = y_train[test_index]

          validation_accuracy, f1_s, y_pred= supportVecMachine(valid_X_test, valid_y_test, valid_X_test, valid_y_test, reg_para, kernel_val, gamma_val)
          print("accuracy on the internal test set", validation_accuracy)

          total_acc += validation_accuracy

        avg_acc = total_acc/10.0

        print("For regularization parameter =", reg_para, ",", kernel_val, ", ", gamma_val, ": average accuracy on the test set is", avg_acc)
      
        if avg_acc > max_acc:
          max_acc = avg_acc
          max_reg_para = [reg_para, kernel_val, gamma_val]
    
  print("The best parameters are", max_reg_para)
  return max_reg_para, y_pred

def svm_stats(X_train, y_train, X_test, y_test, max_reg_para):
  # Trianing on the whole training dataset using the best hyper-parameter
  test_accuracy, f1_s,y_pred = supportVecMachine(X_train, y_train, X_test, y_test, max_reg_para[0], max_reg_para[1], max_reg_para[2])
  print("Accuracy on the testing dataset for SVM:", test_accuracy)
  print("F1 score on the testing dataset for SVM:", f1_s)
  
  return test_accuracy, f1_s,y_pred



"""**Main**"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import scipy.stats as stat

if __name__ == '__main__':
  Xtrain, Ytrain, Xtest, Ytest = loadDataSet()
    
  # Combine the traing set and the testing set
  Xtotal = np.append(Xtrain, Xtest, axis=0)
  Ytotal = np.append(Ytrain, Ytest, axis=0)

  # Preprocessing data, normalize image pixel values to [0, 1]
  Xtotal = normalizeX(Xtotal)

  num_runs = 2
  
  
  
  svm_accuracy_list = []
  svm_f1_list = []
  
  
  
  # Split the total data (280000) into the training set (240000) and the testing set (40000)
  # 6 folds used as the training dataset, 1 fold used as the testing dataset
  skf = StratifiedKFold(n_splits=7, shuffle=True)
  
  for i in range(num_runs):
    # Reshuffle the total number of data (280000) into the training set (240000) and the testing set (40000)
    for train_index, test_index in skf.split(Xtotal, Ytotal):
      # 6 folds used as the training set
      Xtrain = Xtotal[train_index]
      Ytrain = Ytotal[train_index]
      # The remaining fold used as the testing set
      Xtest = Xtotal[test_index]
      Ytest = Ytotal[test_index]
      
      
      
      # Flatten each image
      Xtrain = Xtrain.reshape((Xtrain.shape[0], -1))
      Xtest = Xtest.reshape((Xtest.shape[0], -1))     
      
      
      
      # For SVM
      # Best hyper parameter returned from the cross validation
      svm_best_paras,y_pred = svm_train_cv(Xtrain[:2400], Ytrain[:2400])
      
      # Using the best hyper parameter to train svm using the traing dataset, and return accuracy and f1 score on the testing dataset
      svm_acc, svm_f1 = svm_stats(Xtrain[:2400], Ytrain[:2400], Xtest[:400], Ytest[:400], svm_best_paras)
      svm_accuracy_list.append(svm_acc)
      svm_f1_list.append(svm_f1)
      
      break

  lr_m_acc, lr_lower_acc, lr_upper_acc = mean_confidence_interval(logreg_accuracy_list, 0.95)
  lr_m_f1, lr_lower_f1, lr_upper_f1 = mean_confidence_interval(logreg_f1_list, 0.95)
  
  
  svm_m_acc, svm_lower_acc, svm_upper_acc = mean_confidence_interval(svm_accuracy_list, 0.95)
  svm_m_f1, svm_lower_f1, svm_upper_f1 = mean_confidence_interval(svm_f1_list, 0.95)
  print("For the support vector machine:")
  print("Accuracy:")
  print("Mean:", svm_m_acc, "Lower bound:", svm_lower_acc, "Upper bound:", svm_upper_acc)
  print("F1 score:")
  print("Mean:", svm_m_f1, "Lower bound:", svm_lower_f1, "Upper bound:", svm_upper_f1)

  
  