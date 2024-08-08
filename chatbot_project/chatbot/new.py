import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
intents = json.loads(open('E:/chatbot_project/chatbot/intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Preprocess intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('E:/chatbot_project/chatbot/words.pkl', 'wb'))
pickle.dump(classes, open('E:/chatbot_project/chatbot/classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0:len(words)])
train_y = list(training[:, len(words):])

# Define and train the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('E:/chatbot_project/chatbot/chatbot_model.h5')

print("Model trained and saved.")
