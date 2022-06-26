

class Transaction:
    ''' Represents a transaction, containing some items and a class label.'''
    def __init__(self, items, label, id_):
        ''' Makes a new transaction
        
        Args:
        items: a list of items corresponding to the transaction.
        label: the label corresponding to the transaction.
        id_: A unique id corresponding to the transaction
        '''
        self._items = items
        self._label = label
        self._id    = id_

    def get_items(self):
        '''Returns the list of items for this transaction.

        Returns:
        items of the transaction
        '''
        return self._items

    def get_label(self):
        '''Returns the class label for this transaction.

        Returns:
        class label of the transaction
        '''
        return self._label

    def remove_any(self, to_be_removed):
        s = set(to_be_removed)
        will_be_removed = []
        for item in self._items:
            if item in s:
                will_be_removed.append(item)

        for item in will_be_removed:
            self._remove(item)

    def _remove(self, item):
        self._items.remove(item)


    def __str__(self):
        '''Returns a string showing information about this transaction.

        Returns:
        string showinf items and class label of this transaction.
        '''
        return ', '.join([str(x) for x in self._items]) + ' --> ' + str(self._label)
