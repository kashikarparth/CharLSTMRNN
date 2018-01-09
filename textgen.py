import sys
import numpy
#import tensorflow as tf
from keras.models import load_model
from keras.utils import np_utils
EPOCHS = 0
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
model = load_model('nnmodellarge.h5')
#while(EPOCHS<=10):
#    model = load_model('nnmodellarge.h5')
#    model.fit(X,y,epochs=1,batch_size=128)
#    model.save('nnmodellarge.h5')
#    EPOCHS+=1
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print ("\nDone.")