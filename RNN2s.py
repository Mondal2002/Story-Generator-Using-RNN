import os
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
# Define the path to your text file
txt = "C:/Users/Subham Mondal/Desktop/Subham Mondal/sherlockcompile.txt"
text = open(txt, "r")
data = text.read()
text.close()  # Close the file after reading it

print("Length of data:", len(data))

def clean_text(text):
    """Removes punctuation and non-alphabetic characters from the given text."""
    pattern = re.compile(r"[^\w\s]")
    return pattern.sub("", text)

# Clean the text
cleaned_data = clean_text(data)

cleaned_data = cleaned_data.lower()

charnum = sorted(set(cleaned_data))


chartoidx = dict((c, i) for i, c in enumerate(charnum))
idxtochar = dict((i, c) for i, c in enumerate(charnum))

seq_length = 40
step_size = 3
sentence = []
nextchar = []

# Create sequences and their corresponding next characters
for i in range(0, len(cleaned_data) - seq_length, step_size):
    sentence.append(cleaned_data[i:i+seq_length])
    nextchar.append(cleaned_data[i+seq_length])
'''
X = np.zeros((len(sentence), seq_length, len(charnum)), dtype=np.bool_)
Y = np.zeros((len(sentence), len(charnum)), dtype=np.bool_)

# Vectorize the data
for i, sequences in enumerate(sentence):
    for t, characters in enumerate(sequences):
        X[i, t, chartoidx[characters]] = 1
    Y[i, chartoidx[nextchar[i]]] = 1

# Create and compile the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, len(charnum))))
model.add(Dense(len(charnum)))
model.add(Activation("softmax"))  # "Softmax" should be lowercase

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))  # Use 'learning_rate' instead of 'lr'

# Train the model
model.fit(X, Y, batch_size=256, epochs=4)

# Save the model
model.save("textgen2.model")
'''
import random
def sample(preds,temperature=0.1):
    preds=np.asarray(preds).astype('float64')
    preds=np.log(preds)/temperature
    exp_preds=np.exp(preds)
    preds=exp_preds/np.sum(exp_preds)
    probas=np.random.multinomial(1,preds,1)
    return np.argmax(probas)
model2=tf.keras.models.load_model('textgen2.model')
def generator(length,temperature):
    start_index=10 
    '''random.randint(0,len(cleaned_data)-seq_length-1)'''
    generated=''
    sentence=cleaned_data[start_index:start_index+seq_length]
    generated+=sentence
    for i in range(length):
        X=np.zeros((1,seq_length,len(charnum)))
        for t,character in enumerate(sentence):
            X[0,t,chartoidx[character]]=1
        predictions=model2.predict(X,verbose=0)[0]
        next_index=sample(predictions,temperature)
        next_character=idxtochar[next_index]
        generated +=next_character
        sentence=sentence[1:]+next_character
    return generated
print("\n") 
print("---0.1---\n")
print(generator(500,0.1))

print("---0.2---\n")
print(generator(500,0.2))

print("---0.3---\n") 
print(generator(500,0.3))

print("---0.4---\n") 
print(generator(500,0.4))

print("---0.5---\n")  
print(generator(500,0.5))

print("---0.6---\n")  
print(generator(500,0.6))

print("---0.7---\n")  
print(generator(500,0.7))

print("---0.8---\n")  
print(generator(500,0.8))

print("---0.9---\n")  
print(generator(500,0.9))

print("---1---\n")  
print(generator(500,1))