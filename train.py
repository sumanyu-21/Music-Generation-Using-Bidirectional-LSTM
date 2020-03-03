#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os
import json
import argparse
import model
import numpy as np
#from model import build_model, save_weights
from model import build_model, save_weights


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
# In[16]:


DATA_DIR = './data'
LOG_DIR = './logs'

BATCH_SIZE = 16  # 16 time series at a time.
SEQ_LENGTH = 64


# In[17]:


class TrainLogger(object):
    def __init__(self,file):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = 0
        with open(self.file, 'w') as f:
            f.write('epoch,loss,acc\n')  # COmma Seperated Values(CSV)
    
    def add_entry(self, loss, acc):
        self.epochs+=1
        s='{},{},{}\n'.format(self.epochs,loss,acc)
        with open(self.file, 'a') as f:
            f.write(s);

            
def read_batches(T,vocab_size):
    length = T.shape[0] #129, 665
    batch_chars = int(length/BATCH_SIZE); # 8,104
    
    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH): # (0, 8040 ,64)
        X = np.zeros((BATCH_SIZE, SEQ_LENGTH)) # 16X64
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size)) # 16X64X86
        for batch_idx in range(0, BATCH_SIZE): #(0,16)
            for i in range(0, SEQ_LENGTH): #(0,64)
                X[batch_idx, i] = T[batch_chars*batch_idx + start + i]
                Y[batch_idx,i , T[batch_chars * batch_idx + start + i +1]] = 1;
        yield X,Y



def train(text,epochs = 100, save_freq = 10):
    
#   Character to Index mapping and vice - versa
    char_to_idx = {ch:i for (i, ch) in enumerate(sorted(list(set(text)))) }
    print("Number of Unique characters :" + str(len(char_to_idx))) #86
    
    
#   Creates a new json file and store the dictionary in that json file.
    with open(os.path.join(DATA_DIR, 'char_to_idx.json'), 'w') as f:
        json.dump(char_to_idx, f)
        
    idx_to_char = {i:ch for (ch,i) in char_to_idx.items() }
    vocab_size = len(char_to_idx)
    
# 1) MODEL ARCHITECTURE
    model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size) # This is Building Model architecture
    model.summary()
    model.compile(loss = 'categorical_crossentropy' ,optimizer = 'adam', metrics=['accuracy']) # This is training the model.
#   Minimise cross_entropy loss, as we have multiclass classification problem
#   And we will use accuracy measure as performance metric.
    
    
    
#   Train Data Generation
    T = np.asarray([char_to_idx[c] for c in text], dtype = np.int32) # Changing the entire text from characters to 
                                                                     # Numerical indices.
#   Number of characters in the Text data
    print("Length of Text :" + str(T.size)) # 129, 665
    steps_per_epoch = (len(text)/BATCH_SIZE-1)/SEQ_LENGTH
    
#   Create object of TrainLogger Class
    log = TrainLogger('training_log.csv')
    
    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch+1, epochs))
        
        losses , accs = [], []
        
        # For each iteration in an epoch, we are creating a batch and training the model on batch itself.
        for i, (X,Y) in enumerate(read_batches(T,vocab_size)):
            
            print(X);
            loss, acc = model.train_on_batch(X,Y)
            print('Epoch{}---->Batch {}: loss = {}, acc = {}'.format(epoch, i+1,loss,acc))
            losses.append(loss)
            accs.append(acc)
            
        log.add_entry(np.average(losses), np.average(accs))
        
        if((epoch+1)% save_freq ==0):
            save_weights(epoch+1, model)
            print('Saved checkpoint to', 'weights.{}.h5'.format(epoch+1))
            
    
        


# In[24]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train the model on some text')
    parser.add_argument('--input', default = 'input.txt', help = 'name of the text file to train from')
    parser.add_argument('--epochs', type = int, default = 100, help = 'number of epochs to train for')
    parser.add_argument('--freq', type = int, default = 10, help = 'checkpoint save frequency')
    args = parser.parse_args()
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    train(open(os.path.join(DATA_DIR, args.input)).read(), args.epochs, args.freq)


# In[ ]:





# In[ ]:




