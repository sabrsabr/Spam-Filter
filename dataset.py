import numpy as np
import re

class Dataset:
    def __init__(self, X, y):
        self._x = X # messages 
        self._y = y # labels ["spam", "ham"]
        self.train = None # tuple (X_train, y_train)
        self.val = None # tuple (X_val, y_val)
        self.test = None # tuple (X_test, y_test)
        self.label2num = {'ham': 0, 'spam': 1} # dictionary that converts labels to numbers
        self.num2label = {v:k for k, v in self.label2num.items()} # dictionary that converts numbers to labels
        self._transform()
        
    def __len__(self):
        return len(self._x)
    
    def _transform(self):
        '''
        The function of clearing the message and converting labels to numbers.
        '''
        for i in range(len(self._x)):
            self._x[i] = self._x[i].lower()
            for w in "!@#$%^&*<>?,.'/\�_;:()|-":
                 self._x[i] = self._x[i].replace(w,'')
            self._y[i] = self.label2num[self._y[i]]        
        pass

    def split_dataset(self, val=0.1, test=0.1):
        '''
        A function that splits the dataset into train-validation-test sets.
        '''
        datasets = {}
        np.random.seed(1)
        indices = np.arange(0, len(self))
        np.random.shuffle(indices)
        indices = indices.tolist()
        datasets['train'] = (self._x[indices[int(len(indices)*(val+test)):]], self._y[indices[int(len(indices)*(val+test)):]])
        datasets['val'] = (self._x[indices[:int(len(indices)*val)]], self._y[indices[:int(len(indices)*val)]])
        datasets['test'] = (self._x[indices[int(len(indices)*val):int(len(indices)*(val+test))]], \
                            self._y[indices[int(len(indices)*val):int(len(indices)*(val+test))]])
        self.train = datasets['train']
        self.val = datasets['val']
        self.test = datasets['test']
        return datasets
        pass
