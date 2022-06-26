class Stack:
	
	def __init__(self):
		self.__data = []
	
	def is_empty(self):
		return len(self.__data)==0

	def push(self,item):
		self.__data.append(item)

	def peek(self):
		if self.is_empty():
			raise ValueError
		return self.__data[-1]

	def pop(self):
		if self.is_empty():
			raise ValueError
		del self.__data[-1]

	def __len__(self):
		return len(self.__data)

