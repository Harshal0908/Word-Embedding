from tensorflow.keras.preprocessing.text import one_hot

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