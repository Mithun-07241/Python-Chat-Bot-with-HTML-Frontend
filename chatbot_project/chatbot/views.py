import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import tensorflow as tf
tf.keras.backend.clear_session()

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
intents = json.loads(open('E:/chatbot_project/chatbot/intents.json').read())

words = pickle.load(open('E:/chatbot_project/chatbot/words.pkl', 'rb'))
classes = pickle.load(open('E:/chatbot_project/chatbot/classes.pkl', 'rb'))
model = tf.keras.models.load_model('E:/chatbot_project/chatbot/chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("GO! Bot is running!")

def chatbot(message):
    message = input("You: ")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)
    return res





while True:
    message = input("You: ")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)



