import csv
import re
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
import en_core_web_sm
nlp = en_core_web_sm.load()
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def clean(s):
    s = s.lower()
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s) #punctuation, special characters
    s = re.sub(r'\s+', ' ', s) #whitespace
    s = re.sub(r'https?://\S+', '', s) # remove URLs
    s = re.sub(r'#', '', s) #remove hashtags
    return s

def transform(s):
    doc = nlp(s, disable=['parser', 'ner'])
    lemmas = [token.lemma_ for token in doc]
    a_lemmas = [lemma for lemma in lemmas if lemma.isalpha()]
    return ' '.join(a_lemmas)

train_data['cleantext'] = train_data['text'].apply(lambda s: clean(s))
train_data['cleantext'] = train_data['text'].apply(lambda s: nlp(s))
train_data['cleantext'] = train_data['text'].apply(lambda s: transform(s))
stop_words = set(stopwords.words('english'))
train_data['stopwords'] = train_data['cleantext'].apply(lambda s: 200*len([word for word in s.split() if word in stop_words]))
test_data['cleantext'] = test_data['text'].apply(lambda s: clean(s))
test_data['cleantext'] = test_data['text'].apply(lambda s: nlp(s))
test_data['cleantext'] = test_data['text'].apply(lambda s: transform(s))
test_data['stopwords'] = test_data['cleantext'].apply(lambda s: 200*len([word for word in s.split() if word in stop_words]))

train_data.drop(['text'], axis=1, inplace=True)
x_train = train_data['cleantext']
y_train = train_data['target']
x_test, y_test = x_train, y_train
vectorizer = TfidfVectorizer(stop_words = 'english')
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)
x_predict = test_data['cleantext']
x_predict = vectorizer.transform(x_predict)
x_iterate = test_data['id']

x_train = x_train.toarray()
x_test = x_test.toarray()
x_predict = x_predict.toarray()

x_train = pd.concat([pd.DataFrame(x_train), train_data['stopwords']], axis=1)
x_test = pd.concat([pd.DataFrame(x_test), train_data['stopwords']], axis=1)
x_predict = pd.concat([pd.DataFrame(x_predict), test_data['stopwords']], axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1],)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.selu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.selu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.selu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.selu))
model.add(tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
accuracy, loss = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('disaster.model')

f = open('outputs.csv', 'w')
prediction = model.predict(x_predict)
writer = csv.writer(f)
count = 0
writer.writerow(['id', 'target'])
for i in prediction:
    bucket = []
    bucket.append(np.array(x_iterate[count]))
    if i >= 0.5: bucket.append(1)
    else: bucket.append(0)
    writer.writerow(bucket)
    count += 1
f.close()
