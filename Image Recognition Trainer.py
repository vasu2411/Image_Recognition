from PIL import Image

#image classes
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
          'frog', 'horse', 'ship', 'truck']

from keras.datasets import cifar10

(X_train,y_train),(X_test,y_test) = cifar10.load_data()

from matplotlib import pyplot as plt

from keras.utils import np_utils
new_X_train = X_train.astype('float32')
new_X_test = X_test.astype('float32')
new_X_train /= 255
new_X_test /= 255
new_y_train = np_utils.to_categorical(y_train)
new_y_test = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm

#create model
model = Sequential()

#add layers to images
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, input_shape=(32, 32, 3), activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#model calculates loss and accuracy of itself by predicting each image's category and trained from it
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

#model gets through all the images 10 times for better accuracy
model.fit(new_X_train, new_y_train, epochs=10, batch_size=32)

import h5py
#save trained model
model.save('Trained_model.h5')