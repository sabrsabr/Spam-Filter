import numpy as np
import re

class Model:
    def __init__(self, alpha=1):
        self.vocab = set() # a dictionary containing all unique words from the train 
        self.spam = {} # a dictionary containing word frequencies in spam messages from the train dataset
        self.ham = {} # a dictionary containing the frequency of words in non-spam messages from the train dataset.
        self.alpha = alpha #  smoothing
        self.label2num = None # dictionary used to convert labels to numbers
        self.num2label = None # dictionary used to convert number to labels
        self.Nvoc = None # total number of unique words in the train dataset
        self.Nspam = None # total number of unique words in spam messages in the train dataset
        self.Nham = None # total number of unique words in non-spam messages in the train dataset
        self._train_X, self._train_y = None, None
        self._val_X, self._val_y = None, None
        self._test_X, self._test_y = None, None

    def fit(self, dataset):
        '''
        dataset - an object of the Dataset class
        he function uses the "dataset" input argument,
        to populate all the attributes of the given class.
        '''
        
        self._train_X, self._train_y = dataset.train[0], dataset.train[1]
        self._val_X, self._val_y = dataset.val[0], dataset.val[1]
        self._test_X, self._test_y = dataset.test[0], dataset.test[1]
        
        self.label2num = dataset.label2num
        self.num2label = dataset.num2label
        
        'UNIQUE WORDS self.Nvoc, self.Nspam, self.Nspam'
        for row in dataset.train[0]:
            self.vocab.update(set(row.split())) 
        self.Nvoc = len(self.vocab)
                
        spam_words = set()
        for sam, message in zip(self._train_y, self._train_X):
            if sam == 1:
                spam_words.update(set(message.split()))
        self.Nspam = len(spam_words)
                
        ham_words = set()
        for sam, message in zip(self._train_y, self._train_X):
            if sam == 0:
                ham_words.update(set(message.split()))                
        self.Nham = len(ham_words)   
        
        'FREQUENCY self.spam, self.ham'
        for word in spam_words:
            count = 0
            for sam, message in zip(self._train_y, self._train_X):
                if sam == 1:
                    for i in message.split(' '):
                        if word == i:
                            count += 1
            self.spam[word] = count

        for word in ham_words:
            count = 0
            for sam, message in zip(self._train_y, self._train_X):
                if sam == 0:
                    for i in message.split(' '):
                        if word == i:
                            count += 1
            self.ham[word] = count        
        
        pass
    
    def inference(self, message):
        '''
        The function takes one message and uses a naive bayesian algorithm to identify it as spam/not spam.
        '''
        
        for sign in "'!@#$%^&*()[]{}.,+-=><:\/?-":
            message = message.replace(sign, '')
        message_list = message.lower().split(' ')
        pspam = len(self._train_y[self._train_y==1])/len(self._train_y)
#         print('pspam:', pspam, 'pham:', 1-pspam)
        for word in message_list:
            pspam *= (self.spam.get(word, 0) + self.alpha ) / (sum(self.spam.values()) + self.alpha * self.Nvoc)
         
        pham = 1-pspam
        for word in message_list:
            pham *= (self.ham.get(word, 0) + self.alpha ) / (sum(self.ham.values()) + self.alpha * self.Nvoc)
        
#         print(pspam, pham)

        if pspam > pham:
            return "spam"
        return "ham"
    
    def validation(self):
        '''
        The function predicts message labels from the validation dataset,
         and returns the prediction accuracy of the message labels.
         You must use the inference() class method.
        '''
        success = 0
        fail = 0
        for sam, message in zip(self._val_y, self._val_X):
            if self.inference(message) == self.num2label[sam]:
                success += 1
            else:
                fail += 1
                print('fail #', fail, message)
        val_acc = success / (success + fail)

        return val_acc 

    def test(self):
        '''
        The function predicts message labels from the test dataset,
        and returns the prediction accuracy of the message labels.
        You must use the inference() class method.
        '''

        success = 0
        fail = 0
        for sam, message in zip(self._test_y, self._test_X):
            if self.inference(message) == self.num2label[sam]:
                success += 1
            else:
                fail += 1
                print('fail #', fail, message)
        test_acc = success / (success + fail)

        return test_acc


