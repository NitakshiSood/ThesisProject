from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
#import chiTest
from sklearn.model_selection import StratifiedKFold


def NeuralNetwork(X_train,X_test,y_train,y_test):
    le = preprocessing.LabelEncoder()
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #y_train = scaler.transform(y_train)
    #y_test= scaler.transform(y_test)
    num_runs=2
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    max_acc = 0.0
    best_params = []

    best_folds=[]
    best_scores=0
    for i in range(num_runs):
        # Reshuffle the total number of data (280000) into the training set (240000) and the testing set (40000)
        for train_index, test_index in skf.split(X_train, y_train):
            Xtrain = X_train[train_index]
            Ytrain = [y_train[j] for j in train_index]
                #[a[:, :j] for j in i]
            # The remaining fold used as the testing set
            Xtest = X_train[test_index]
            Ytest = [y_train[j] for j in test_index]

            best_folds=nn_train_cv(Xtrain,Ytrain)

            mlp = MLPClassifier(hidden_layer_sizes=best_folds[0], max_iter=best_folds[1],
                                activation=best_folds[2], solver='lbfgs')
            mlp.fit(Xtest, Ytest)
            predictions = mlp.predict(Xtest)

            best_scores=mlp.score(X_test,y_test)

            if(best_scores>max_acc):
                max_acc=best_scores
                best_params=best_folds[0:3]

            print("The best parameters are: ", best_params)
            print("accuracy is: ", max_acc)
    mlp = MLPClassifier(hidden_layer_sizes=best_params[0], max_iter=best_params[1], activation=best_params[2], solver='lbfgs')
    mlp.fit(X_test, y_test)  # .values.ravel()

    predictions = mlp.predict(X_test)
    print("Accuracy of testing using NN: ",mlp.score(X_test,y_test))
    #chiTest.calcChiSqr(y_test, predictions)
    '''
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    from sklearn.utils.fixes import signature

    precision, recall, _ = precision_recall_curve(score[max_acc][3], score[max_acc][4])

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    '''

def nn_train_cv(X_train, y_train):
    #hidden_layer_sizes = [(100, 100), (150), (150, 100, 100), (200, 150), (200)]
    max_iter = [1000, 500, 2000]
    activation = ['logistic', 'relu']
    hidden_layer=(len(X_train)+len(y_train))//2
    print(hidden_layer)
    hidden_layer_sizes=[]
    hidden_layer_sizes.append(hidden_layer)
    hidden_layer_sizes.append(hidden_layer+10)
    hidden_layer_sizes.append(hidden_layer-10)
    hidden_layer_sizes.append((hidden_layer,hidden_layer))

    score = {}
    best = {}

    # 10-fold internal stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in skf.split(X_train, y_train):
        for hidden in hidden_layer_sizes:
            for iter in max_iter:
                for acc in activation:
                    Xtrain = X_train[train_index]
                    Ytrain = [y_train[j] for j in train_index]
                    # The remaining fold used as the testing set
                    Xtest = X_train[test_index]
                    Ytest = [y_train[j] for j in test_index]

                    # Flatten each image
                    Xtrain = Xtrain.reshape((Xtrain.shape[0], -1))
                    Xtest = Xtest.reshape((Xtest.shape[0], -1))

                    mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=iter, activation=acc, solver='lbfgs')
                    mlp.fit(Xtrain, Ytrain)  # .values.ravel()

                    predictions = mlp.predict(Xtest)
                    # print(type(predictions))
                    # print("ytest shape ",y_test.shape," predicted shape ",predictions.shape," xtrain shape ",X_train.shape," xtest shape ",X_test.shape)
                    # print(confusion_matrix(y_test,predictions))
                    # print(classification_report(y_test,predictions))
                    # print("hidden layers: ",hidden," iterations",iter," activation", acc)
                    # print(mlp.score(X_test,y_test))
                    score[mlp.score(Xtest, Ytest)] = [hidden, iter, acc, Ytest, predictions]

    max_acc = max(score.keys())

    return score[max_acc]








'''
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

                    validation_accuracy, f1_s = supportVecMachine(valid_X_test, valid_y_test, valid_X_test, valid_y_test, reg_para, kernel_val, gamma_val)
                    #print("accuracy on the internal test set", validation_accuracy)

                    total_acc += validation_accuracy

                avg_acc = total_acc/10.0

                #print("For regularization parameter =", reg_para, ",", kernel_val, ", ", gamma_val, ": average accuracy on the test set is", avg_acc)

                if avg_acc > max_acc:
                    max_acc = avg_acc
                    max_reg_para = [reg_para, kernel_val, gamma_val]

    print("The best parameters are", max_reg_para)
    return max_reg_para
'''