#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import dependencies
import numpy
import sys
#import nltk (if error in tokenize_words)
#nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


# In[2]:


#load data
file= open('Frankenstein.txt').read()


# In[3]:


#tokenisation of text
#replace some words with tokens

#convert text to lower case- standerdisation
def tokenize_words(input):
    input=input.lower()
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(input)
    filtered= filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)

processed_inputs= tokenize_words(file)
#tokenization done


# In[4]:


#charactes to numbers
chars=sorted(set(processed_inputs))
char_to_num = dict((c,i) for i, c in enumerate(chars))


# In[5]:


#check if char to num has worked
input_len=len(processed_inputs)
vocab_len= len(chars)
print('No. of characters=', input_len)
print('No. of vocabs=', vocab_len)


# In[6]:


#sequence length
seq_len= 100
xdata=[]
ydata=[]


# In[7]:


#loop thro the seq
for i in range(0, input_len-seq_len, 1):
    in_seq=processed_inputs[i:i +seq_len]
    out_seq= processed_inputs[i+ seq_len]
    xdata.append([char_to_num[char] for char in in_seq])
    ydata.append([char_to_num[char] for char in out_seq])

npatterns=len(xdata)
print('Total patterns=',npatterns)


# In[8]:


#input sequence to np array
x=numpy.reshape(xdata, (npatterns, seq_len, 1))
x=x/float (vocab_len)


# In[9]:


#one-hot encoding
y=np_utils.to_categorical(ydata)


# In[10]:


#creating a seq model
model= Sequential()
model.add(LSTM(256, input_shape= (x.shape[1], x.shape[2]), return_sequences= True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences= True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))


# In[11]:


#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[12]:


#saving weights
filepath= 'model_weights_saved.hdf5'
checkpoint= ModelCheckpoint(filepath, monitor='loss', verbose= 1, save_best_only=True, mode='min')
desired_callbacks=[checkpoint]


# In[13]:


#fit model and let it train
model.fit(x, y, epochs=1, batch_size= 256, callbacks= desired_callbacks)
#increase no of epochs to increase accuracy
#epochs=100


# In[1]:


#recompiling model with saved weights
filename='model_weights_saved.hdf5'
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[15]:


#output of the model into chars
num_to_char= dict( (i,c) for i, c in enumerate(chars))


# In[ ]:


#random seed to help generate
start= numpy.random.randint(0, len(xdata)-1)
pattern= xdata[start]
print("Random seed: ")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")


# In[ ]:


#generate the text using model
for i in range(1000):
    x= nmpy.reshape(pattern, (1, len(pattern), 1))
    x=x/float(vocab_len)
    prediction=model.predict(x, verbose=0)
    index= numpy.argmax(prediction)
    result= num_to_char[index]
    seq_in=[num_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern=pattern[1:len(pattern)]
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




