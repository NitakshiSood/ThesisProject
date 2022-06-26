import os
import sys
import time
import random
import pandas as pd
import numpy as np

from associative_classifier import AssociativeClassifier

def run_test(fold_number,input_name,test_name):
    sep = ' '
    global g_rules_count, np_rules_count
    g_rules_count = 0
    np_rules_count = 0

    clf = AssociativeClassifier()
    t1, t2 = clf.fit(file_name=input_name, sep=sep, fold_number=fold_number)
    g_rules_count += t1
    np_rules_count += t2
    
    # Now let's test the model.
    # we need to read the test file first,
    # then we need to separate the labels from it.
    # and then run the algorithms on the given items
    # finally we check them with the given labels
    with open(test_name, 'r') as f:
        data = f.read().strip().split('\n')
    dxy = [line.strip().split(sep) for line in data]
    
    
    ''' collecting features from dataset'''
    features_int = set(x for line in dxy for x in line[:-1])
    columns = [str(x) for x in features_int] + ['class']
    
    row_dicts = list(map(lambda row: {**{k:1 for k in set(row[:-1])}, **{'class':int(row[-1])}}, dxy))
    df = pd.DataFrame(row_dicts, columns=columns)
    df = df.fillna(0).astype(int)

    accs = [0,0,0]
    
    max_accs=0
    for hrs in [1,2,3]:
        predictions = clf.predict(df, hrs)
        
        accs[hrs-1] = df[df['class'].astype(int)==predictions.astype(int)].shape[0]
        if(max_accs<accs[hrs-1]):
            max_accs = accs[hrs-1]
            predictions_best=predictions
        print('accuracy S'+str(hrs)+' is:', accs[hrs-1]/df.shape[0])   

    return list(map(lambda x:x/df.shape[0], accs)),predictions_best


if __name__ == '__main__':
    ''' Note:
    to understand and separate different item types (original, and my ordered version)
    original should be string since it is read from file, and it must be categorical at the 
    end of the day. And thus mine can be integer.\n1 \n1 \n1 \n1 \n1 \n1 \n1 \n1 \n1 \n1 
    '''

def main(dataset_name,counter):
    acc_acc = []
    #dataset_name = sys.argv[1]
    sep = ' '
    test_count = 1

    g_rules_count = 0
    np_rules_count = 0

    #random.seed(1)
    #np.random.seed(1)

    t0 = time.time()
    # pr.enable()
    
    #for fold_number in range(1,test_count+1):
    #for fold_number in range(1,test_count+1):    
        #input_name  = 'datasets/' + dataset_name + '_train1'+ '.txt'
        #test_name   = 'uci/' + dataset_name + '_ts' + str(fold_number) + '.txt'
    input_name  = 'datasets_new_boosting/' + dataset_name + '_train'+ str(counter)+'.txt'
    test_name   = 'datasets_new_boosting/' + dataset_name + '_test' + str(counter) + '.txt'
    print("KITNI BAR?")
    print(counter)
    accs,predictions = run_test(str(counter),input_name,test_name)
    acc_acc.append(accs)
    print('\n\n')
    sys.stdout.flush()
    print('average generated rules count:',g_rules_count/test_count)
    print('average not pruned rules count:',np_rules_count/test_count)
    for hrs in [1,2,3]:
        print('S'+str(hrs)+':',dataset_name, '--total accuracy:', sum([x[hrs-1] for x in acc_acc])/test_count)

    print('total time:',int(time.time()-t0))
    # pr.disable()
    # pr.print_stats(sort='time')
    return predictions