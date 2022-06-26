from decimal import *
getcontext().prec = 5
from scipy.special import gammaln
from config import Configuration as config
import util
from math import pow,log10,isnan, log2, frexp, log, exp
import math
from sys import getsizeof
from util import my_lcomb
from array import array
import numpy as np
import math
import scipy.stats
#from fisher import pvalue
from scipy.stats import hypergeom
import functools

@functools.lru_cache(maxsize=8192)
def lgamma(x):
    '''
    if x<0: # just make this change for anneal and breast
        x=-x
    elif x==0:
        x=0.1
        '''
    return math.lgamma(x)

class RuleNode:
    ''' Represents a node for each rule (aka, each Node has a RuleNode object 
    for each of labels it has seen.)
    '''
    __slots__ = '_arr_bool', '_arr_num'
    _comb = util.my_comb
    _fact = util.my_factorial
    _log  = util.my_log
    def __init__(self):
        # count, _min_ss, _ss
        self._arr_num          = [0.0, 1.0, 1.0]# array('d', [0.0, 1.0, 1.0])
        # is_pss, _is_minimal
        self._arr_bool         = array('h', [0,0])
    def del_unnecessary(self):
        pass

    # getters
    def get_count(self):
        #return self._count
        return self._arr_num[0]

    def get_ss(self):
        #return self._ss
        return self._arr_num[2]

    #def get_pss(self):
    #    return self._pss

    def get_is_pss(self):
        return self._arr_bool[0]==1

    def get_is_ss(self):
        return self._arr_num[2]<=config.ALPHA

    def get_is_minimal(self):
        return self._arr_bool[1]==1

    def get_min_ss(self):
        #return self._min_ss
        return self._arr_num[1]

    # setters
    def increase_count(self):
        #self._count += 1.0
        self._arr_num[0]+=1.0

    def set_pss(self, antecedent_support, label_support, d_size):
        if antecedent_support > label_support:
            #self._is_pss = True
            self._arr_bool[0] = 1
        else:
            temp = np.array([d_size - antecedent_support + 1, label_support + 1, d_size + 1, label_support - antecedent_support + 1])
            temp2 = gammaln(temp)
            temp = temp2[0] + temp2[1] - temp2[2] - temp2[3] 
            self._arr_bool[0] = 1 if temp <= -3.0 else 0

    def set_ss(self, antecedent_support, label_support, d_size, subsequent_support):
        min_n = min(int(antecedent_support), int(label_support)) - int(subsequent_support)
        intersection = subsequent_support
        union = (antecedent_support+label_support) - intersection

        lz = lgamma(d_size  + 1) - lgamma(label_support  + 1) - lgamma(d_size - label_support + 1);

        t1 = lgamma(antecedent_support + 1);
        t2 = lgamma(d_size - antecedent_support  + 1);

        sum_ = 0.0
        l1 = 0
        l2 = 0
        temp = np.zeros(min_n+1)

        def f(i):
            l1 = t1 - lgamma(subsequent_support + i  + 1) - lgamma(antecedent_support - subsequent_support - i  + 1);
            l2 = t2 - lgamma(d_size - union + i  + 1) - lgamma(d_size - antecedent_support - d_size + union - i  + 1);
            return Decimal(l1+l2-lz)

        temp_f = np.vectorize(f)
        temp = temp_f(range(min_n+1))
        sum_ = np.exp(temp).sum()

        self._arr_num[2] = Decimal(sum_)


    def set_minimal(self, antecedent_support):
        self._arr_bool[1] = 1 if self.get_count() >= antecedent_support else 0
        
    def set_min_ss(self, parent_ss):
        self._arr_num[1] = min(self._arr_num[2], parent_ss)

    def __str__(self):
        return str(self.get_count()) + \
           " " +    str(self.get_is_pss()) + \
           " " +    str(self.get_is_ss()) + \
           " " +    str(self.get_is_minimal()) + \
           " " +    "_ss:" + \
           " " +    str(self._arr_num[2]) + \
           " " +    "_min_ss:" + \
           " " +    str(self._arr_num[1])

