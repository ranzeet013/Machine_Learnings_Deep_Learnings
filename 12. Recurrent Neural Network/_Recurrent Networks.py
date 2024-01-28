#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()


# In[4]:


lens = [len(x) for x in X]
sns.distplot(lens)
print(np.percentile(lens, 90))


# In[3]:


vocab_size = 10000
embedding_dim = 50 

(x, y), _ = imdb.load_data(num_words=vocab_size)


# In[6]:


from keras.preprocessing import sequence

maxlen = 500
x = sequence.pad_sequences(x, maxlen=maxlen)


# Creating an RNN model sequentially starting with an embedding layer, the vocabulary size and embedding dimensions. LSTM layer is added with the embedding dimension as its input size. Lastly, a dense layer with a single neuron and a sigmoid activation function is included. 

# In[7]:


from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential

RNN = Sequential([
    Embedding(vocab_size, embedding_dim), 
    LSTM(embedding_dim), 
    Dense(1, activation='sigmoid')
])

RNN.summary()


# Compiling the RNN model with the optimizer as 'adam', the loss function as 'binary_crossentropy', and the metric for evaluation as accuracy ('acc'). Training the model using the `fit` method with input data `x` and corresponding labels `y`, running for 5 epochs and utilizing a validation split of 10%. 

# In[9]:


RNN.compile(optimizer='adam', loss='binary_crossentropy', 
            metrics=['acc'])

history = RNN.fit(x, y, epochs=5, 
                  validation_split=0.1)


# In[ ]:




