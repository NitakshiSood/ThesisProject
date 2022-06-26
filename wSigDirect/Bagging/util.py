from functools import reduce
#import numpy as np
#from scipy.special import comb
from config import *
import math
import sys
from math import *
import operator as op

_log_dict  = dict()
_comb_dict = dict()
_lcomb_dict = dict()
_fact_dict = {0:1}

for i in range(1,100000):
	_log_dict[i] = math.log(i)

for i in range(1,10000):
	_fact_dict[i] = i*_fact_dict[i-1]

"""
import operator as op
def comb(n, r):
    #r = int(min(r, n-r))
    #r = int(r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer//denom

from operator import mul    # or mul=lambda x,y:x*y
from fractions import Fraction
#from operator import op
import operator as op
def comb(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    #denom = reduce(op.mul, range(1, r+1), 1)
    denom = my_factorial(r)
    return numer//denom
"""

def my_comb2(n,c):
	#return 2 ** int((my_lgamma(n)-my_lgamma(c)-my_lgamma(n-c))/math.log(2))
	return math.pow(math.e, (math.lgamma(n+1)-math.lgamma(c+1)-math.lgamma(n-c+1)))
	#return 2 ** int(my_log(my_factorial(n))-my_log(my_factorial(c))-my_log(my_factorial(n-c)))

def my_lcomb(n,c):
	if (n,c) in _lcomb_dict:
		return _lcomb_dict[(n,c)]

	temp = math.lgamma(n+1)-math.lgamma(c+1)-math.lgamma(n-c+1)
	_lcomb_dict[(n,c)] = temp
	return temp


def my_comb(n, c):
	#return comb(n,c)
	try:
		if n==0 or c==0 or n==c:
			return 1
		#n = max(n1,c1)
		#c = min(n1,c1)
		c = min(c,n-c)
		#print(n,c)
		if (n,c) in _comb_dict:
			return _comb_dict[(n,c)]
		#if (n, n-c) in _comb_dict:
		#	return _comb_dict[(n,n-c)]
		#val = comb(n,c)
		### ??? if I comment the small cases, number of generated rules will change!
		if (n,c-1) in _comb_dict:
			val = _comb_dict[(n,c-1)] //(c) * (n-c+1) 
			_comb_dict[(n,c)] = val
			return val
		if (n,c+1) in _comb_dict:
			val = _comb_dict[(n,c+1)] //(n-c) * (c+1) 
			_comb_dict[(n,c)] = val
			return val

		if n<10:
			val = (my_factorial(n)//my_factorial(c))//my_factorial(n-c)
		else:
			#val = comb(n,c)
			val = my_comb2(n,c)
			#print("good!")
		_comb_dict[(n,c)] = val
		return val
	except Exception as e:
		#print('exception in combination')
		return float('inf')


def my_comb_log(n1,c1):
	if n1==0 or c1==0 or n1==c1:
		return 0
	n = max(n1,c1)
	c = min(n1,c1)
	#if (n,c) in _comb_dict:
	#	return _comb_dict[(n,c)]
	#if (n, n-c) in _comb_dict:
	#	return _comb_dict[(n,n-c)]
	#val = comb(n,c)
	### ??? if I comment the small cases, number of generated rules will change!
	#if n<10:
	#	val = (my_factorial(n)//my_factorial(c))//my_factorial(n-c)
	#else:
	#	val = my_comb2(n,c)
	#_comb_dict[(n,c)] = val

	return int(my_log(my_factorial(n))-my_log(my_factorial(c))-my_log(my_factorial(n-c)))

"""
def my_lgamma(n):
	if n in _lgamma_dict:
		return _lgamma_dict[n]
	val = lgamma(n)
	print(val)
	_lgamma_dict[n] = val
	return val
"""
def my_factorial(n):
	if n in _fact_dict:
		return _fact_dict[n]
	if (n-1) in _fact_dict:
		_fact_dict[n] = _fact_dict[n - 1]*n
		return _fact_dict[n]
	if n>0 and (n+1) in _fact_dict:
		_fact_dict[n] = _fact_dict[n + 1]//n
		return _fact_dict[n]

	_fact_dict[n] = math.factorial(n)
	#_fact_dict[n] = (lgamma(n+1)//log(10))
	return _fact_dict[n]


def my_log(n):
	n = int(n)
	if n in _log_dict:
		return _log_dict[n]
	if n<<1  in _log_dict:
		_log_dict[n] = _log_dict[n<<1]-1
		return _log_dict[n]
	if n>>1 in _log_dict:
		_log_dict[n] = 1 + _log_dict[n>>1]
		return _log_dict[n]
	else:
		_log_dict[n] = math.log2(n)
		return _log_dict[n]