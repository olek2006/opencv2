import pandas as pd #таблиці
import numpy as np #математичні операції
import tensorflow as tf #для побудови моделей та їх тренування
from tensorflow import keras #бібліотека для тс, апі для неї
from tensorflow.keras import layers #побудова шарів нейкронки
from sklearn.preprocessing import LabelEncoder #перетворює текстові мітки в числа
import matplotlib.pyplot as plt #для побудови графіків

df = pd.read_csv('data/figures.csv')
#print(df)
df['area_to_perimeter_ratio'] = df['area'] / df['perimeter']
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

#елементи для навчання
X = df[['area', 'perimeter', 'corners','area_to_perimeter_ratio']] #ознаки
y = df['label_enc']
#створюэмо модель
model = keras.Sequential([layers.Dense(8, activation = 'relu', input_shape = (4,)),
                          layers.Dense(16, activation = 'relu'),
                          layers.Dense(8, activation = 'softmax')])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#навчання моделі
history = model.fit(X,y, epochs = 500, verbose = 0)

#візуалізація навчання
plt.plot(history.history['loss'], label = 'Втрата (Loss)')
plt.plot(history.history['accuracy'], label = 'Точність (Accuracy)')
plt.xlabel("Епоха")
plt.ylabel("Значення")
plt.title('Процес навчання')
plt.legend()
plt.show()

#тестування
test = np.array([[20, 18, 3, 1.111]])

pred = model.predict(test)
print(f'Імовірність по кожному з класів: {pred}')
print(f'Модель визначила:{encoder.inverse_transform([np.argmax(pred)])}')





