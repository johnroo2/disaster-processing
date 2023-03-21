import csv
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import gensim

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def getsim(sentence, md):
    sum = 0
    count = 0
    for j in gensim.utils.simple_tokenize(sentence):
        count += 1
        sum += md.wv.similarity(j, 'disaster')
    return sum/count

def normalize(df):
    words = []
    for i in df['text']:
        temp = []
        for j in gensim.utils.simple_tokenize(i):
            temp.append(j)
        words.append(temp)
    md = gensim.models.Word2Vec(words, min_count = 1,
                              vector_size = 100, window = 5)
    df['length'] = df['text'].apply(lambda text: len(text.split()))
    df['disaster'] = df['text'].apply(lambda sentence: getsim(sentence, md))
    df['disaster'] = (df['disaster']-df['disaster'].mean())/df['disaster'].std()
    df.drop(['keyword', 'location', 'text'], axis=1, inplace=True)
    return df

y_train = train_data['target']
x_train = normalize(train_data)
x_train.drop(['target', 'id'], axis=1, inplace=True)
x_test, y_test = x_train, y_train
x_iterate = normalize(test_data)
x_predict = x_iterate.copy()
x_predict.drop(['id'], axis=1, inplace=True)

print(x_train)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape = (2, )))
model.add(tf.keras.layers.Dense(units=16, activation=tf.nn.selu))
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.selu))
model.add(tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=25)
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
    bucket.append(np.array(x_iterate['id'][count]))
    if i >= 0.4: bucket.append(1)
    else: bucket.append(0)
    writer.writerow(bucket)
    count += 1
f.close()


