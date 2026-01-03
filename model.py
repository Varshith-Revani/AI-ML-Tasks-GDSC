import numpy as np
import tensorflow as tf
import matplotlib as plt
import cv2
import os 

#loading the dataset from tf
mnist = tf.keras.datasets.mnist

#Spliting training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Noramalise pixel brightness to (0,1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Creating Model
model = tf.keras.models.Sequential()

# Adding layers
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model/ train
model.fit(x_train, y_train, epochs =3)

c = 'yes'
while c == 'yes':
    n =int(input("select the image number from uploaded images: "))
    img = cv2.imread(f"digits/Untitled{n}.png")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"This digit is probably : {np.argmax(prediction)}")
    c =input("Do you want to try again (yes/no) :")





