#!/usr/bin/env python3

import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score

import sigdirect

class _Preprocess:

    def __init__(self):
        self._label_encoder = None

    def preprocess_data(self, raw_data):
        """ Given one of UCI files specific to SigDirect paper data,
            transform it to the common form used in sklearn datasets"""
        transaction_data =  [list(map(int, x.strip().split())) for x in raw_data]
        max_val = max([max(x[:-1]) for x in transaction_data])
        X,y = [], []

        for transaction in transaction_data:
            positions = np.array(transaction[:-1]) - 1
            transaction_np = np.zeros((max_val))
            transaction_np[positions] = 1
            X.append(transaction_np)
            y.append(transaction[-1])
        X = np.array(X)
        y = np.array(y)

        # converting labels
        if self._label_encoder is None:# train time
            unique_classes = np.unique(y)
            self._label_encoder = defaultdict(lambda: 0, zip(unique_classes, range(len(unique_classes))))
            y = np.vectorize(self._label_encoder.get)(y)
        else:# test time
            y = np.vectorize(lambda a: self._label_encoder[a])(y)

        return X,y

def test_uci(dataset_name,counter,input_name,test_name):

    #assert len(sys.argv)>1
    #dataset_name = sys.argv[1]
    print(dataset_name)
    '''
    if len(sys.argv)>2:
        start_index = int(sys.argv[2])
    else:
        start_index = 1
    '''
    start_index=1
    final_index = 1
    k = final_index - start_index + 1

    all_pred_y = defaultdict(list)
    all_true_y = []

    ####
    # for trains set
    all_pred_y_train = defaultdict(list)
    all_true_y_train = []
    avg_train = [0.0] * 4
    
    ####
    # counting number of rules before and after pruning
    generated_counter = 0
    final_counter     = 0
    avg = [0.0] * 4

    tt1 = time.time()

    for index in range(start_index, final_index +1):
        #print(index)
        prep = _Preprocess()

        # load the training data and pre-process it
        train_filename = input_name#os.path.join('uci', '{}_tr{}.txt'.format(dataset_name, index))
        with open(train_filename) as f:
            raw_data = f.read().strip().split('\n')
        X,y = prep.preprocess_data(raw_data)
        clf = sigdirect.SigDirect(get_logs=sys.stdout)
        generated_c, final_c = clf.fit(X, y)
        
        generated_counter += generated_c
        final_counter     += final_c

        ############################################
        #ADDING THIS FOR TRAIN SET SCORES->
        
        # evaluate the classifier using different heuristics for pruning
        for hrs in [3]:#(1,2,3):
            y_pred_train = clf.predict(X, hrs)
            print('TRAIN ACC S{}:'.format(hrs), accuracy_score(y, y_pred_train))
            avg_train[hrs] += accuracy_score(y, y_pred_train)

            all_pred_y_train[hrs].extend(y_pred_train)
            y_prob_train=clf.predict_proba(X, hrs)
            #print("y_prob_train: ",y_prob_train)
            #print(stop)

                
        ############################################

        
        # load the test data and pre-process it.
        test_filename  = test_name#os.path.join('uci', '{}_ts{}.txt'.format(dataset_name, index))
        with open(test_filename) as f:
            raw_data = f.read().strip().split('\n')
        X,y = prep.preprocess_data(raw_data)

        # evaluate the classifier using different heuristics for pruning
        hrs_acc={}
        max_acc=0        
        for hrs in [3]:#(1,2,3):
            y_pred = clf.predict(X, hrs)
            print('TESTING ACC S{}:'.format(hrs), accuracy_score(y, y_pred))
            avg[hrs] += accuracy_score(y, y_pred)
            acctemp =accuracy_score(y, y_pred)
            hrs_acc[hrs]=acctemp
            if acctemp>max_acc:
                max_acc=acctemp
            
            all_pred_y[hrs].extend(y_pred)
            y_prob=clf.predict_proba(X, hrs)
            #print("y_prob: ",y_prob)
            #print(stop)

        all_true_y.extend(list(y))
        print('\n\n')
    
    
    #print("y_prob: ",y_prob)
    #print("y_prob_train: ",y_prob_train)
    #print("type pf y_prob_train: ",type(y_prob_train))
    import pandas as pd
    #creating numpy array:
    y_prob=pd.DataFrame(data=y_prob).T
    y_prob_train=pd.DataFrame(data=y_prob_train).T
    
    #print("y_prob: ",y_prob)
    #print("y_prob_train: ",y_prob_train)
    #print("type pf y_prob_train: ",type(y_prob_train))    
    #print()
    #print(stophere)
        
    print(dataset_name)
    
    #for hrs in [2]:#(1,2,3):
    #    print('AVG ACC S{}:'.format(hrs), accuracy_score(all_true_y, all_pred_y[hrs]))
    
    bestacc=accuracy_score(all_true_y, all_pred_y[hrs])
    print('INITIAL RULES: {} ---- FINAL RULES: {}'.format(generated_counter/k, final_counter/k))
    print('TOTAL TIME:', time.time()-tt1)
    
    #predictions, acc,g_rules_count,np_rules_count,hrs_acc,scores,scores_testset
    return y_pred, bestacc,generated_counter/k,final_counter/k,hrs_acc,y_prob_train,y_prob#,scores_testset
    

#if __name__ == '__main__':
def main(dataset_name,counter,input_name,test_name):
    y_pred, bestacc,gen_count,final_count,hrs_acc,y_prob_train,y_prob=test_uci(dataset_name,counter,input_name,test_name)
    #print("y_pred: ",y_pred)
    #print("bestacc: ",bestacc)
    #print("gen_count: ",gen_count)
    #print("final_count: ",final_count)
    #print("y_prob: ",y_prob)
    #print(stop)
    return y_pred, bestacc,gen_count,final_count,hrs_acc,y_prob_train,y_prob
