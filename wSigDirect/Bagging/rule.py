import math

class Rule:
	""" Represents a final Rule that is extracted from trie. """
	def __init__(self, items, label, confidence, ss, support):
		self._items = items
		self._label = label
		self._confidence = confidence
		self._ss = ss
		self._support = support

	def get_items(self):
		return self._items

	def get_label(self):
		return self._label

	def get_confidence(self):
		return self._confidence

	def get_ss(self):
		return self._ss

	def get_support(self):
		return self._support

	def __str__(self):
		try:
			return ' '.join([str(x) for x in self._items]) + ' ' + \
							str(self._label) + ';' + str(format(self._support,'.4f')) + \
							',' + str(format(self._confidence,'.3f')) + ',' + \
							str(format(self._ss.ln(), '.3f'))
		except:
			return ' '.join([str(x) for x in self._items]) + ' ' + \
							str(self._label) + ';' + str(format(self._support,'.4f')) + \
							',' + str(format(self._confidence,'.3f')) + ',' + \
							str(0)
