from rule_node import RuleNode
from array import array

class Node:
    ''' Represents a node in the apriori-like trie
    It gives access to its children via children field,
    and to its different rule_nodes via labels_dict
    '''
    __slots__ = 'children', '_count', '_rule_nodes', '_labels'
    def __init__(self):
        # a dict that for each label, keeps an object that holds some numbers for them
        # (like ss value, and etc). Labels are added only if we have seen such a thing in our dataset.
        #self.labels_dict = {}
        self._rule_nodes = []
        self._labels = array('H')
        # link to children nodes
        self.children    = {}
        # number of itemsets of this path found in dataset
        # float to limit memory usage
        self._count = 0.0

    #def get_pss_labels(self):
    #    all_pss_labels = []
    #    for label, rule_node in self.labels_dict.items():
    #        if rule_node.get_is_pss():
    #            all_pss_labels.append(label)

    def get_count(self):
        return int(self._count)

    def increase_count(self):
        self._count += 1.0

    def get_children(self):
        return self.children

    def get_child(self, item):
        return self.children[item]

    def has_child(self, item):
        return item in self.children

    def add_child(self, item):
        self.children[item] = Node()

    def has_label(self, label_):
        return self._labels.count(int(label_))>=1

    def add_rule_node(self, label_):
        self._labels.append(int(label_))
        self._rule_nodes.append(RuleNode())

    def get_rule_node(self, label_):
        place = self._labels.index(int(label_))
        #print(label_, self._labels, place)
        return self._rule_nodes[place]

    def get_labels_size(self):
        return len(self._rule_nodes)

    #def get_labels(self):
    #    return list(map(str,self._labels.tolist()))

    def get_label_rule_nodes(self):
        #print(list(zip(self._labels, self._rule_nodes)))
        return list(zip([str(x) for x in self._labels.tolist()], self._rule_nodes))

    def remove_label(self, label_):
        index = self._labels.index(int(label_))
        self._labels.pop(index)
        del self._rule_nodes[index]