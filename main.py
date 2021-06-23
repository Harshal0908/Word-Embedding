from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np

sent = ['The glass of milk',
        'the glass of juice',
        'the cup of Tea',
        'I am a good boy',
        'I am a good developer',
        'Understan the meaning of words',
        'your videos are good',]

print(sent)

voc_size = 10000

onehot_repr = [one_hot(words,voc_size) for words in sent]
print(onehot_repr)

sent_len = 8
embedded_docs = pad_sequences(onehot_repr,padding = 'pre',maxlen=sent_len)
print(embedded_docs)
