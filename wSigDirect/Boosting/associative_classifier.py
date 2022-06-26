#import objgraph
import gc
import itertools
import sys
import time
import random
import numpy as np
import math
#from scipy.special import comb
import pandas as pd

import database
from stack import Stack
from node import Node
from rule_node import RuleNode
from rule import Rule
from config import *
from transaction import Transaction


class AssociativeClassifier:
    #c = Combination()
    #_comb = c.comb
    def __init__(self):
        '''instanciates needed variables for an associative classifier
        The main task is done in the fit function.
        '''
        # A dictionary that for each label, keeps all rules in a list.
        # it is used in the classification part.
        self._label_rules_dict = None

    def _parse_input_args(self,kwargs):
        '''This function parses the input arguments
        to train an associative classifier

        '''
        print('kwargs:',kwargs)

        if 'file_name' in kwargs:
            self._file_name = kwargs['file_name']
        else:
            print('no file_name!')
            sys.exit(1)
            
        if 'sep' in kwargs:
            self._sep = kwargs['sep']
        else:
            self._sep = None
        if 'index' in kwargs:
            self._index = kwargs['index']
        else:
            self._index = None

        if 'fold_number' in kwargs:
            self._fold_number = kwargs['fold_number']
            
        
        if 'dataset_name' in kwargs:
            self.dataset_name = kwargs['dataset_name']
        else:
            print('no file_name!')
            sys.exit(1)    
        #self._file_name = file_name
        
    def __str__(self):
        #print('_database', self._database)
        print('_id_item_dict', self._id_item_dict)
        print('_item_id_dict',sorted(self._item_id_dict.items(),key=lambda x:int(x[0])))
        print('_ordered_item_ids',self._ordered_item_ids)
        print("\nprinting tree: item_id, (item_name), count, labelset:" + 
                    " {label: count, pss, ss, non-redundant, minimal}\n")
        return str(self.traverse_print().strip())

    # These functions are not needed anymore
    def _get_rule_confidence(self, rule):
        ''' This function returns the confidence value for a given rule.
        Note that this only is used for pruning section and when the tree
        is built.

        Args:
        rule: a rule that is generated in the generation phase.
            It should be a tuple that the first item is the rule items
            where items are integers, and the second one should be the 
            label which is a string

        Returns:
        the confidence score for the given rule 0 if it doesn't exist.

        '''
        # iterating over items of the rule:
        if rule is None or len(rule)==0 or len(rule[0])==0:
            return 0
        label = rule[1]
        curr = self._root_node
        for r in rule[0]:
            if r not in curr.children:
                return 0.0
            curr = curr.children[r]
        return curr.labels_dict[label].get_count() / curr.get_count()

    def _get_pf_lower_bound(self, x_support, ck_support):
        ''' This function computes the lower bound of pf-value using
        corollary 2. If support of X is greater than support of ck,
        0 is returned, else, the value is computed based on the given formula

        Args:
        x_support: support value for x
        ck_support: support value for ck label.

        Returns: lower bound of pf-value based on corollary 2.
        '''
        if x_support<ck_support:
            d_size = self._database.get_database_size()
            return (math.factorial(d_size-x_support)*math.factorial(ck_support))/\
                    (math.factorial(d_size)*math.factorial(ck_support - x_support))
        else:
            return 0.0

    ################## train section ##################
    def fit(self, **kwargs):
        self._parse_input_args(kwargs)
        self._database = database.TransactionalDatabase(self._file_name,
                                                self._index, 
                                                self._sep,
                                                use_corollary_1=False) #my change set use corollary to true, originally it was false
        #SET THIS VALUE
        #WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_v7_10bins_coded
        #dataset_name='breast'#heart_tobetransformed_converted #WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_v6_CODED'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType2_FS1_SMOTE'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_30bins_v4'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_v5_30bins_transfored'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType2_FS1_SMOTE_10bins'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_v5_30bins_transfored'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_30bins'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_v5_30bins_transfored'#'WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_v5_30bins_transfored'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS4_nodiscretization'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_noMissing_30bins'#WATT_Data_March14_2018_green_no_assessment_positive_train_correctType_FS1_30bins'#WATT2018_12Features_Prep (1)_discretized_30bins'
        #str(counter)='1' IS SET HERE
        
        self._valfile_name='datasets_new_boosting/' + self.dataset_name + '_validation'+ self._fold_number +'.txt'
        print("readin validation file named: ",self._valfile_name)
        
        #self._valfile_name='datasets_new_boosting/' + dataset_name + '_validation'+ '1' +'.txt'
        
        #self._valfile_name='datasets_new_boosting/' + dataset_name + '_validation'+ '1' +'.txt'
        self._valdatabase = database.TransactionalDatabase(self._valfile_name,
                                                self._index, 
                                                self._sep,
                                                use_corollary_1=False)        
        self._all_labels_size = len(self._database.get_all_labels())
        ### Corollary 1
        # the pruning of corollary 1 should be done over the database object
        # (in the next step, I am assuming it is already done.)
        ### now, let's convert them to ids here, and sort them,
        ### whenever that I want to get items, I should use id_item dict data
        self._make_ordered_items()
        # Now, we will build the tree!
        # layer by layer, and for each node, if it is not PSS,
        # do not continue that. And if it is, get the exact number(?)
        
        self._converted_transactions = list(self._convert_transaction(x) for x in self._database.get_transactions())
        #print("###########################################self._converted_transactions: ##############################")
        #print(self._converted_transactions)
        
        self._generated_rules = self._build_tree()

        self._generated_rules = self.back_to_original_id(self._generated_rules)

        print('~~~~~~~~~~~~~~~~~~~~~~~~generated rules count:', len(self._generated_rules))
        print('generated rules: ')
        #for rule in self._generated_rules:
        #    print(str(rule))        
        
        with open(self._file_name + '.rules', 'w') as f:
            #print("here?")
            for rule in self._generated_rules:
                #print("YES")
                f.write(str(rule))
                f.write('\n')
        all_rules_count = len(self._generated_rules)
    
        # let's prune the rules that are not needed
        # using Algorithm 2
        np_rules_count = self._prune_rules()
        print('~~~~~~~~~~~~~~~~~~~~~~~~pruned rules count:', len(self._generated_rules))
        print("~~~~~~~~~~~~~~~~~~~~~~~~np_rules_count : ",np_rules_count)
        print('pruned rules: ')
        #for rule in self._generated_rules:
        #    print(str(rule))        
        with open(self._file_name + 'afterpruning'+'.rules', 'w') as f:
            #print("here?")
            for rule in self._generated_rules:
                print(str(rule))
                f.write(str(rule))
                f.write('\n')        
        return all_rules_count, np_rules_count

    def traverse_print(self, current_node=None, separator='   ', tab_count = 0):
        if current_node is None:
            current_node = self._root_node
        # This is true only for root_node
        if current_node.item==None:
            me = 'ROOT\n'
        else:
            me =  separator * tab_count + str(current_node.item) + \
                    '(' + str(self._id_item_dict[current_node.item]) + ')' + \
                    str(current_node.get_count()) + \
                    ' LABELSET: ' + str([(x,str(y)) for x,y in sorted(current_node.labels_dict.items())])
            x_support = current_node.get_count()
            pfs = ''
            for label in sorted(current_node.labels_dict, key=lambda x:int(x)):
                pfs+= ' ' + str(label) + ' ' + str(current_node.labels_dict[label].get_ss())
            me += pfs + '\n'
        kids = []
        for child_node in current_node.children.values():
            kids.append(self.traverse_print(child_node, separator, tab_count+1))
        return me + ''.join(kids)
                
    ### making ordered items ###
    def _make_ordered_items(self):
        ''' First, sorts all items based on their count in the databse
        and then generates unique ids for each of them. (reversed order)
        It keeps two dictionaries that maps each id to item and vice versa.
        Second, it will keep the order of items in a separate list.
        '''
        # A dictionary that maps an id to the corresponding item.
        self._id_item_dict = dict()
        # A dictionary that maps an item to its corresponding id.
        self._item_id_dict = dict()
        # A list of ids that are ordered based on their support count.
        self._ordered_item_ids = []

        # To make the ids more informative, ids will be built based on the
        # support counts, so that the _ordered_item_ids are ordered numbers.
        #print('from database, ', self._database.get_sorted_items_list())
        for item in self._database.get_sorted_items_list():
            new_id = self._generate_id()
            if new_id in self._id_item_dict:
                raise ValueError("The id already exists in the dictionary!")
            self._id_item_dict[new_id] = item
            if item in self._item_id_dict:
                raise ValueError("The item already exists in the dictionary")
            self._item_id_dict[item]   = new_id
            self._ordered_item_ids.append(new_id)

    def _generate_id(self):
        ''' This function generates a new id and returns
        it once it is called.
        '''
        if '_AssociativeClassifier__generator' not in self.__dict__:
            self.__generator = 0
        self.__generator += 1
        return self.__generator


    ### tree creation ###
    def _build_tree(self):
        ''' This function makes the apriori-like trie. Each node is 
            represented using an instance of Node class.
            This process id done step by step, and the output of this function
            is equivalent to the end of Rule Generation phase ofthe algorithm.
            This one is based on BFS and includes prunning, (similar to apriori)
        '''
        created_all_rules = []
        self._root_node = Node()
        depth = 0
        created_rule_nodes_count = 0
        tt0 = time.time()
        
        while depth==0 or created_rule_nodes_count>0:
            tt1 = time.time()
            depth += 1
            self._deepen_tree(depth)
            tt2 = time.time()
            print('(deepen done) depth increased to',depth, 'TIME:', tt2-tt1)
            sys.stdout.flush()
            created_rule_nodes_count, created_rules = self._set_ss_values(depth)
            created_all_rules.extend(created_rules)
            tt3 = time.time()
            print('(set_ss done) ss values are set...', 'total final rules so far:', len(created_all_rules), 'TIME:', tt3-tt2)
            sys.stdout.flush()

            # calling garbage collector explicitly
            if created_rule_nodes_count>200000:
                gc.collect(generation=2)
                print('Garbage Collector')
                sys.stdout.flush()
        
        print("fold TIME:", time.time()-tt0)
        return created_all_rules
 


    def _deepen_tree(self, depth):
        ''' This function traverses the trie and extends it by one layer.
        To do that, it has to scan the database once and for nodes that 
        we know they are statistically significant (done in the previous step)
        it will add children. For such nodes, it will 1. see if they are PSS
        if so, it will compute whether they are statistically significant, 
        and finally compute and see if they are minimal.

        '''
        for (id_items, label) in self._converted_transactions:
            self._step_traverse(self._root_node, 0, id_items, label, 0, depth)
        

    def _convert_transaction(self, transaction):
        temp = [self._item_id_dict[x] for x in transaction.get_items()]
        #items_id = tuple(sorted(temp, key= lambda x:self._ordered_item_ids.index(x)))
        items_id = list(sorted(temp))
        label = transaction.get_label()
        return items_id, label


    def _step_traverse(self, node, start_index, id_items, label, curr_layer, max_depth):
        '''
        Args:
        node: The current node we are in it.
        id_items: A list of items corresponding to the itemset (ids)
        label: The class label for the corresponding itemset
        curr_layer: The curr_layer that this recursive function is in.
        max_depth: the layer that we want to build now
            (and later run PSS check, etc)
            if max_depth is one, we should create root's children since 
                root itself already exists!

        '''
        ''' traverses the trie recursively.
        If we are in a non-leaf node(curr_layer<max_layer-1), 
        if a node for the next item exists,
        it will move to that node, else , discontinue that item in that branch.
        If we are in a leaf(curr_layer=max_layer-1), 
        creates the next layer (if still doesn't exist),
        and updates the count and etc. (max_layer)
        We will check the PSS, exact pscore, and minimality of the node later.
        '''
        # we need to create new nodes in this layer.
        #if len(id_items)==0:
        #    return
        
        if (len(id_items)-start_index)<max_depth - curr_layer:
            return

        # this rule_node has been pruned earlier and thus doesn't exist.
        #if curr_layer>0 and label not in node.labels_dict:
        #    #print('THIS IS PRUNING!')
        #    return

        # Let's check if we are still allowed to go deeper:
        # This ensures that up to this item, we were SS.
        #if curr_layer>0 and node.labels_dict[label].get_count()<0:
        #    print('This never happens!')
        #    return
        
        # node is minimal, so no more rules, and not going deper!
        #if curr_layer>0 and node.labels_dict[label].get_is_minimal():
        #    return

        #if curr_layer>0 and not node.labels_dict[label].get_is_pss():
        #    return
        def my_f(item):
            if item not in node.children:
                node.children[item] = Node()
            child = node.children[item]
                #print('size of node.children', sys.getsizeof(node.children[item]))
            if not child.has_label(label):
            #if label not in node.children[item].labels_dict:
                child.add_rule_node(label)#labels_dict[label] = RuleNode()
            child.get_rule_node(label).increase_count()
            child.increase_count()

        # we are on the max_layer, let's create its children
        if curr_layer==max_depth-1:
            list(map(my_f, id_items[start_index:]))
            return
        else:
            for i in range(start_index, len(id_items)):
                item = id_items[i]
                if item in node.children:
                    self._step_traverse(node.children[item], i+1, id_items, label, curr_layer+1, max_depth)


    def back_to_original_id(self, rules):
        new_rules = []
        for rule in rules:
            temp = sorted([ float(self._id_item_dict[x]) for x in rule.get_items()],key=lambda x:float(x))
            new_rule = Rule(temp,rule.get_label(), rule.get_confidence(), rule.get_ss(), rule.get_support())
            new_rules.append(new_rule)
        return new_rules


    ### ss values computation ###

    def _set_ss_values(self, depth):
        ''' This function traverses the trie and for all leaf nodes, 
        it makes sure that all rules (X-->c_k) are statistically significant, 
        otherwise, their corresponding value in the labels_dict will be set 
        to -1.
        It first checks if they are PSS, and if so, computes the pf-value 
        is computed.

        '''
        # number of rules added at the end of this level
        self.rcounter = 0

        def my_f(p):
            inc = p[0].get_labels_size()#len(p[0].labels_dict)
            dec, rules = self._remove_non_pss_rule_nodes_and_return_rules(p[0], depth, p[1], p[2], p[3])
            self.rcounter += (inc-dec)
            return rules


        stack  = Stack()
        curr_depth = 0
        parent = None
        parental_item = None
        stack.push((self._root_node, [], curr_depth, parent, parental_item))
        inc, dec = 0, 0
        # keep all nodes in last level
        last_level_nodes = []
        while not stack.is_empty():
            node, items, curr_depth, parent, parental_item = stack.peek()
            stack.pop()
            
            # add nodes of last level to a list, to compute ss values faster
            if curr_depth==depth:
                last_level_nodes.append((node, items, parent, parental_item))
            # just traverse!
            else:
                for item, child in node.children.items():
                    stack.push((child, items + [item], curr_depth+1, node, item))
        ll = map(my_f, last_level_nodes)
        all_rules = list(itertools.chain.from_iterable(ll))
        print('totally added', self.rcounter, 'rules', end='\t')
        sys.stdout.flush()
        return self.rcounter, all_rules



    def _is_rule_pss_and_not_minimal(self, items, label, ignore_index):
        curr = self._root_node
        seen = 0
        for i,item in enumerate(items):
            if i==ignore_index:
                seen = 1
                continue
            if (item not in curr.children) or ((i-seen)>0 and not curr.has_label(label)):#label not in curr.labels_dict):
                return False, 1.0
            curr = curr.children[item]
        if not curr.has_label(label):# not in curr.labels_dict:
            return False, 1.0
        label_node = curr.get_rule_node(label)#labels_dict[label]
        if label_node.get_is_pss() and (not label_node.get_is_minimal()):
            return True, label_node.get_min_ss()
        return False, label_node.get_min_ss()

    def _are_all_parents_pss_and_not_minimal_and_min_pss(self,all_items, label):
        min_all_ss = 1.0
        for ignore_index in range(len(all_items)):
            pss_not_minimal, pss= self._is_rule_pss_and_not_minimal(all_items, label, ignore_index)
            min_all_ss = min(min_all_ss, pss)
            if not pss_not_minimal:
                return False, min_all_ss
        return True, min_all_ss


    def _set_rule_node_ss_values(self, rule_node, label, antecedent_support, d_size):
        label_support = self._database.get_label_support(label)
        subsequent_support = int(rule_node.get_count())
        rule_node.set_pss(antecedent_support, label_support, d_size)
        if rule_node.get_is_pss():
            rule_node.set_ss(antecedent_support, label_support, d_size, subsequent_support)
        rule_node.set_minimal(antecedent_support)



    def _remove_non_pss_rule_nodes_and_return_rules(self, node, level, items, parent, parental_item):
        all_counter = 0
        rules = []
        d_size = self._database.get_database_size()
        antecedent_support = node.get_count()

        def my_g(pair):
            counter = 0
            label = pair[0]
            rule_node = pair[1]
            if level==1:
                self._set_rule_node_ss_values(rule_node, label, antecedent_support, d_size)
                if rule_node.get_is_ss():
                    r = Rule(items, str(label), 
                            rule_node.get_count()/node.get_count(), 
                            rule_node.get_ss(), 
                            rule_node.get_count()/d_size)
                    rules.append(r)
                    if rule_node.get_is_minimal():
                        node.remove_label(label)#labels_dict[label]
                        counter += 1
                        return counter
                elif not rule_node.get_is_pss():
                    node.remove_label(label)#labels_dict[label]
                    counter += 1
                    return counter
                if node.has_label(label):# in node.labels_dict:
                    rule_node.set_min_ss( rule_node.get_ss())

            else:
                parents_pss_not_minimal, min_ss = self._are_all_parents_pss_and_not_minimal_and_min_pss(items, label) 
                if not (parents_pss_not_minimal):# and node.labels_dict[label].get_ss()<min_ss):
                    node.remove_label(label)#labels_dict[label]
                    counter += 1
                    return counter
                self._set_rule_node_ss_values(rule_node, label, antecedent_support, d_size)
                if not rule_node.get_is_pss():
                    node.remove_label(label)#labels_dict[label]
                    counter += 1     
                    return counter      
                if rule_node.get_is_ss() and (rule_node.get_ss()<min_ss):# and node.labels_dict[label].get_is_non_redundant():
                    r = Rule(items, str(label), 
                            rule_node.get_count()/node.get_count(), 
                            rule_node.get_ss(), 
                            rule_node.get_count()/d_size)
                    rules.append(r)
                    if rule_node.get_is_minimal():
                        node.remove_label(label)#labels_dict[label]
                        counter += 1
                elif rule_node.get_is_pss():
                    # just keep it for children
                    # remove unneccessary fields of the rule_node
                    #rule_node.del_unnecessary()
                    pass
                if node.has_label(label):# in node.labels_dict:
                    #min_ss = min(min_ss, rule_node.get_ss())
                    rule_node.set_min_ss(min_ss)

            return counter


        all_counter = sum(map(my_g, node.get_label_rule_nodes()))
        if node.get_labels_size()==0:
            del parent.children[parental_item]
        return all_counter, rules



    ### pruning ###
    def _prune_rules(self):
        ''' Using Algorithm 2, we want to know which rules should be pruned
        and which ones not. For each transaction, we find the rule that matches 
        it and has the highest confidence score, then we increase the "count"
        for that rule by 1 (this count is kept in a dictionary)

        Args:
        None

        Returns:
        number of remaining rules
        '''
        
        print("????????before  PRUNING len(_generated_rules): ",len(self._generated_rules))
        #self.pruneripper()
        print("????????AFTER Ripper PRUNING len(_generated_rules): ",len(self._generated_rules))        
        rules_dict = dict()        
        #for transaction in self._database.get_transactions():
        for transaction in self._valdatabase.get_transactions():    
            
            label = transaction.get_label()
            items = transaction.get_items()
            #print("items: ",items)
            #print("label : ",label)
            transaction_items_set = set([float(x) for x in items])
            max_confidence = -1.0
            max_rule = None
            for rule in sorted(self._generated_rules, key=lambda rule: str(rule.get_items())): # try using this for sorting with confidence values
                if rule.get_label()!=label:
                    continue
                if all([item in transaction_items_set for item in rule.get_items()]):
                    confidence = rule.get_confidence()
                    if confidence>max_confidence:
                        max_confidence = confidence
                        max_rule =  rule
            if max_rule is None:
                # print('max_rule is None!! transaction was:', items, '-->', label)
                continue
            if max_rule in rules_dict:
                rules_dict[max_rule] += 1
            else:
                rules_dict[max_rule]  = 1
        #print("Rule dict: ",rules_dict) #it is of the form <RULE OBJECT: NUMBER>, .....
        '''
        import collections
        print("NOW this thing: ")
        new_rules_dict=rules_dict
        from collections import defaultdict
        #new_rules_dict = defaultdict(list)
        new_rules_dict = {}
        for k in rules_dict:
            print("K: ",k)
            print("v: ",rules_dict[k])
            new_rules_dict[k]=[rules_dict[k],k.get_confidence()]
            #new_rules_dict[k].append(k.get_confidence())
        
        print("Now check the new_rules_dict: ")
        print(new_rules_dict)
        #od = collections.OrderedDict(sorted(rules_dict.keys.__str__()))
        #for k, v in od.items(): print(k, v)
        new_rules_dict=dict(sorted(new_rules_dict.items(), key=lambda e: e[1][1],reverse=True))
        
        print("after sorting Now check the new_rules_dict: ")
        print(new_rules_dict)        

        '''
        self._generated_rules = rules_dict.keys()
        self.rulecount=rules_dict
        #print("????????before  PRUNING len(_generated_rules): ",len(self._generated_rules))
        #self.pruneripper()
        print("rules_dict: ",rules_dict)
        
        #print("????????AFTER Ripper PRUNING len(_generated_rules): ",len(self._generated_rules))
        return len(self._generated_rules) #as per new pruning strategy
        #return len(rules_dict) #original
    
    def update_condifence(self, _rule_dict, _transactions):
        #print("heyyy")
        #print(len(_transactions))
        for rule in _rule_dict:
            numerator= denomerator = 0
            for transaction in _transactions:
                if all([item in transaction[0] for item in rule.get_items()]):
                    denomerator += 1
                    if rule.get_label() == transaction[1]:
                        numerator += 1
            if denomerator == 0:
                _rule_dict[rule] = 0
            else:
                _rule_dict[rule] = float(numerator/denomerator)
            
        #print("_rule_dict: ")        
        #print(_rule_dict)
        #print(len(_rule_dict))
        return _rule_dict
            
         
         
    def get_high_conf_rule(self, _rule_dict):
        max_conf=0
        selected_rule = 0
        for rule in _rule_dict:
            if max_conf< _rule_dict[rule]:
                max_conf = _rule_dict[rule]
                selected_rule = rule
 
        return (selected_rule, max_conf)
         
            
    def pruneripper(self):
        _generated_rules = self._generated_rules
        _rule_dict = {}
        for rule in _generated_rules:
            _rule_dict[rule]  = rule.get_confidence()
        
        _rule_dict = dict(sorted(_rule_dict.items(), key=lambda e: e[1],reverse=True))
        _transactions = []
        #print("Before RULE DICT _rule_dict: ",_rule_dict)
        print("Before RULE DICT _rule_dict lenght: ",len(_rule_dict))
        
        #for transaction in self._database.get_transactions():
        for transaction in self._valdatabase.get_transactions():
            label = transaction.get_label()
            items = transaction.get_items()    
            _transactions.append([[float(item) for item in items], label])
        
        pruned_rules = []
        #while not is_empty(_rule_dict):
        while _rule_dict and _transactions: # try using this for sorting with confidence values
              # for transactions
            selected_rule, conf = self.get_high_conf_rule(_rule_dict) 
            #if not conf:
            #    break
            print("conf: ",conf)
            if not conf or conf < 0.50: #0.50
                break            
            #print("selected_rule: ",selected_rule)
            #print("confidence", conf)
            pruned_rules.append(selected_rule) #these are the selected rules after pruning
            for transaction in _transactions:    
                #print("transaction[0] : ",transaction[0] )
                #print("selected_rule.get_items(): ",selected_rule.get_items())
                
                #print(sdfsdfdsg)
                if all([item in transaction[0] for item in selected_rule.get_items()]): # We are not sure to check the label (for now we do check)
                    #print("transaction[1]: ",transaction[1])
                    #print("selected_rule.get_label(): ",selected_rule.get_label())
                    #if selected_rule.get_label()!=transaction[1]:
                    #_transactions.remove(transaction) #try1
                    #try 2
                    
                    if selected_rule.get_label()==transaction[1]: #try2 thoughts
                        #print("I am removing transaction")
                        _transactions.remove(transaction)
                    
                    #original
                    ''' 
                    if selected_rule.get_label()!=transaction[1]: #original thoughts
                        #print("I am removing transaction")
                        _transactions.remove(transaction)
                    '''
                 #   continue
                #if all([item in transaction_items_set for item in rule.get_items()]):        
        
            _rule_dict = self.update_condifence(_rule_dict, _transactions)
            del _rule_dict[selected_rule]
        
        self._generated_rules = pruned_rules
        # TODO : put a treshold in case that confidence get very low
        #for rule in _generated_rules:
        #    print("AM I HERE??????:",str(rule))
        #print("????????AFTER Ripper PRUNING len(_generated_rules): ",len(self._generated_rules))
        #print("AFTER RULE DICT _rule_dict: ",len(_rule_dict))
        #print("_transactions: ", len(_transactions))
        #print("pruned_rules: ",len(pruned_rules))
        #print(weigh)
        #print(asdbhjhbds)
        return 
    def _newprune_rules(self): #NOT USEFUL
        rules_dict = dict()        
        #for transaction in self._database.get_transactions():
        print("################type of self._generated_rules: ",type(self._generated_rules))
        
        #print(sorted(self._generated_rules, key=lambda rule: str(rule.get_confidence()))) #str(rule.get_confidence()))
        
        #for rule in self._generated_rules:
        #    print("rule: ",str(rule))        
        
        #for rule in self._generated_rules:
        print("self._valdatabase: ",self._valdatabase)
        print("type of : self._valdatabase : ",type(self._valdatabase))
        # net_transac_set=[] #ccreate a  temp transaction set which is removed one
        transaction_list=[]
        #transaction=Transaction(self, items, label, id_)
        id_counter=0
        for transaction in self._valdatabase.get_transactions():    
            
            print("transaction: ",transaction)
            print("type(transaction): ",type(transaction))
            
            # Let's convert each item in the transaction to the 
            # ids that we have generated for them.
            #id_items, label = self._convert_transaction(transaction)
            #items, label = self._convert_transaction(transaction)
            label = transaction.get_label()
            items = transaction.get_items()
            print("items: ",items)
            print("label : ",label)
            transaction_items_set = set([int(x) for x in items])
            max_confidence = -1.0
            max_rule = None
            #print("sorted(self._generated_rules, key=lambda rule: str(rule.get_items())): ")
            #print(sorted(self._generated_rules, key=lambda rule: str(rule.get_items())))
            
            for rule in sorted(self._generated_rules, key=lambda rule: str(rule.get_items())):
                print("str(rule.get_items()): ",str(rule.get_items()))
                print("rule: ",rule)
                print("str(rule): ",str(rule))
                #print(weigh)                
                if rule.get_label()!=label:
                    continue
                if all([item in transaction_items_set for item in rule.get_items()]):
                    print("transaction_items_set",transaction_items_set)
                    print("label",label)
                    print("id_counter",id_counter)
                    #transaction=Transaction(self, str(transaction_items_set), label, id_counter)
                    #id_counter=id_counter+1
                    transaction_list.append([transaction_items_set,label])
                    print("transaction_list: ",transaction_list)
                    print(weigh)
                    confidence = rule.get_confidence()
                    if confidence>max_confidence:
                        max_confidence = confidence
                        max_rule =  rule
            if max_rule is None:
                # print('max_rule is None!! transaction was:', items, '-->', label)
                continue
            if max_rule in rules_dict:
                rules_dict[max_rule] += 1
            else:
                rules_dict[max_rule]  = 1
        self._generated_rules = rules_dict.keys()
        #self._newprune_rules()
        return len(rules_dict)        
        
    ################## test section ##################
    def predict(self, x, heuristic=1):
        ''' Given a list of instances, predicts their corresponding class
        labels and returns the labels.

        Args:
        x: a pandas dataframe
        heuristic: the heuristic used in classification (S1, S2, S3)

        Returns:
        a list of labels corresponding to all instances.
        '''
        
        # This should be done only once
        if self._label_rules_dict is None:
            self._make_label_rules_dict()
            
            
        # adding this step to make the classifier more weak by the approach discusses on 29th oct.    
        #print("self._label_rules_dict: ",self._label_rules_dict)
        print("*********************************************")
        weakclassifier_ruledict = dict()
        count1=0
        for k, v in self._label_rules_dict.items():
            print("key: ",k)
            print("v : ",v)
            print("type of v: ",type(v))
            for value in v:
                print(str(value)) 
                count1+=1
        #print(weiighhhhh)
        for label in self._label_rules_dict.keys():
            counter=0
            for rule in self._label_rules_dict[label]:
                if counter>15:
                    break
                print("rule: ",str(rule))
                print("rule conf: ",rule.get_confidence())   
                if(rule.get_confidence() > 0.50):                
                #if(rule.get_confidence() == 1.00):
                    counter+=1
                    print("***************************")
                    if rule.get_label() not in weakclassifier_ruledict:
                        weakclassifier_ruledict[rule.get_label()] = [rule]
                    else:
                        weakclassifier_ruledict[rule.get_label()].append(rule)                    
        #print(weiighhhhh)
        count2=0
        for k, v in weakclassifier_ruledict.items():
            #print("key: ",k)
            for value in v:
                #print(str(value)) 
                count2+=1
        print("initial length: ",len(self._label_rules_dict),count1)        
        print("length afterwards :",len(weakclassifier_ruledict),count2) 
        self._label_rules_dict=weakclassifier_ruledict
        count1=0
        for k, v in self._label_rules_dict.items():            
            for value in v:
                count1+=1        
        print("initial length: ",len(self._label_rules_dict),count1)
        #print(stophere)
        #ends
        
        
        
        self._hrs = heuristic

        if type(x) == np.ndarray:
            predictions = np.apply_along_axis(self._predict_instance, axis=1, arr=x)
        elif type(x) == pd.core.frame.DataFrame:
            predictions = x.apply(self._predict_instance, axis=1)   
        elif type(x) == np.array:
            predictions = self._predict_instance(x)
        elif type(x) == list:
            predictions = self._predict_instance(x)
        else:
            raise Exception("Invalid data type detected in predict_proba")
        
        return predictions

    def predict_proba(self, x, heuristic=1): #i think this is not called
        ''' Given a list of instances, predicts their corresponding class
        labels and returns the labels.

        Args:
        x: a pandas dataframe
        heuristic: the heuristic used in classification (S1, S2, S3)

        Returns:
        a list of labels corresponding to all instances.
        '''

        # This should be done only once
        if self._label_rules_dict is None:
            self._make_label_rules_dict()

        self._hrs = heuristic

        if type(x) == np.ndarray:
            prediction_probs = np.apply_along_axis(self._predict_proba_instance, axis=1, arr=x)
        elif type(x) == pd.core.frame.DataFrame:
            prediction_probs = x.apply(self._predict_proba_instance, axis=1)   
        elif type(x) == np.array:
            prediction_probs = self._predict_proba_instance(x)
        elif type(x) == list:
            prediction_probs = self._predict_proba_instance(x)
        else:
            raise Exception("Invalid data type detected in predict_proba")

        return prediction_probs

    def _predict_instance(self, instance, hrs=None):
        ''' Given an instance, predicts its class label
        Args:
        instance: one instance 
        hrs: the heuristic used to classify it.

        Returns:
        class label corresponding to the instance
        '''
        if hrs is None:
            hrs = self._hrs
        # removing features that are not available in this instance.
        if type(instance) in [pd.core.series.Series, pd.core.frame.DataFrame]:
            instance =  instance[instance==1].index.astype(float)
        elif type(instance) in [np.ndarray, list]:
            instance = np.where(np.asarray(instance)==1)[0]
        # Now, for each label, compute the corresponding score.
        scores = list(map(lambda x: (self._get_similarity_to_label(instance, x, hrs), x), 
                                    sorted(self._label_rules_dict)))
        # find best score based on heuristic
        return self._get_best_match_label(scores, hrs)

    def _predict_proba_instance(self, instance, hrs=None):
        ''' Given an instance, predicts its class label
        Args:
        instance: one instance 
        hrs: the heuristic used to classify it.

        Returns:
        a probability score for each class using softmax
        TODO: probability scores are not good!
        '''
        
        if hrs is None:
            hrs = self._hrs

        # removing features that are not available in this instance.
        if type(instance) in [pd.core.series.Series, pd.core.frame.DataFrame]:
            instance =  instance[instance==1].index.astype(float)
        elif type(instance) in [np.ndarray, list]:
            instance = np.where(np.asarray(instance)==1)[0]

        # Now, for each label, compute the corresponding score.
        scores = list(map(lambda x: (self._get_similarity_to_label(instance, x, hrs)), sorted(self._label_rules_dict)))
        # softmax 
        # making sure they are all _positivize_scores
        scores = self._positivize_scores(scores, hrs)
        # TODO, needs to be resolved
        #scores = list(map(math.exp, scores))
        s = sum(scores)
        scores = [x/s for x in scores]
        return scores

    def _make_label_rules_dict(self):
        ''' Given a list of rules(that contains a label in it for each)
        make a dictionary that the keys are labels, and values are
        rules for each label
        '''
        self._label_rules_dict = dict()
        for rule in self._generated_rules:
            if rule.get_label() not in self._label_rules_dict:
                self._label_rules_dict[rule.get_label()] = [rule]
            else:
                self._label_rules_dict[rule.get_label()].append(rule)

    def _get_similarity_to_label(self, instance, label, hrs):
        
        if hrs==1:
            f = self._hrs_1
        elif hrs==2:
            f = self._hrs_2
        elif hrs==3:
            f = self._hrs_3

        sum_ = 0.0
        #print("self._label_rules_dict[label]: ",self._label_rules_dict[label])
        for rule in self._label_rules_dict[label]:
            if self._rule_matches(instance, rule):
                sum_ += f(instance, rule) * self.rulecount[rule]

        return sum_

    def _rule_matches(self, instance, rule):
        instance_items_set = set(instance)
        for id_item in rule.get_items():
            if id_item not in instance_items_set:
                return False
        return True

    def _get_best_match_label(self, scores, hrs):
        ''' Given a list of pair of score-label, decides the best label
        based on the heuristic given.

        Args:
        scores: a list of score-label pairs
        hrs: heuristic used

        Returns:
        best matching label
        '''

        min_ = min(scores, key=lambda x:(x[0],x[1]))
        max_ = max(scores, key=lambda x:(x[0],x[1]))
        #print("????? hrs used: and scores: ",hrs,scores)
        #print("min_: ",min_)
        #print("max_: ",max_)
        # these heuristics look for minumum score
        
        if hrs in [1,3]:            
            return min_[1]
        else:
            return max_[1]
        
        
              
        

    def _positivize_scores(self, scores, hrs):
        ''' Compute absolute values

        Args:
        scores: a list of scores
        hrs: heuristic used

        Returns:
        scores all positive and highest as the best
        '''
        return [abs(x) for x in scores] if scores else None

    def get_applicable_rules(self, instance):
        '''

        Returns:
        A list of label-rules pair where for each label,
        we get a list of Rule objects that 'label' is its consequent.
        (the list for a label can be empty) 
        '''
        # removing features that are not available in this instance.
        if type(instance) in [pd.core.series.Series, pd.core.frame.DataFrame]:
            instance =  instance[instance==1].index.astype(int)
        elif type(instance) in [np.ndarray, list]:
            instance = np.where(np.asarray(instance)==1)[0]
        all_applicable_rules = []
        for label in sorted(self._label_rules_dict):
            applicable_rules = []
            for rule in self._label_rules_dict[label]:
                
                if self._rule_matches(instance, rule):
                    applicable_rules.append(rule)
            all_applicable_rules.append((label, applicable_rules))
        return all_applicable_rules

    def _hrs_1(self, instance, rule):
        x = rule.get_ss()
        if x> 2*-500:
            #print(" ~~~~~~~~~~!!!!!! HRS 1")
            return float(x.ln())
        else:
            #print(" ~~~~~~~~~~~!!!!!! HRS 1")
            return -float('inf')

    def _hrs_2(self, instance, rule):
        #print(" ~~~~~~~~~~~~!!!!!! HRS 2")
        return rule.get_confidence()

    def _hrs_3(self, instance, rule):
        # return math.log(rule.get_ss()) * rule.get_confidence()
        x = rule.get_ss()
        #print(" !!!!!! HRS 3")
        if x> 2*-500:
            return float(x.ln()) * rule.get_confidence()
        else:
            return -float('inf')
