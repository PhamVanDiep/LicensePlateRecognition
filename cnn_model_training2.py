from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from load_dataset import load_dataset
from keras.utils import np_utils
import tensorflow as tf
import numpy as np

(X_train, Y_train), (X_test, Y_test), (X_val, Y_val) = load_dataset()

num_classes = 36

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_val = np_utils.to_categorical(Y_val, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

callbacks = [
    EarlyStopping(
        monitor='val_acc', 
        patience=36,
        mode='max',
        verbose=1),
    ModelCheckpoint('cnn_model_training1.h5',
        monitor='val_acc', 
        save_best_only=True, 
        mode='max',
        verbose=0)
]

# input_shape = (28, 28, 1)
filter_pixel = 3
droprate = 0.25 #can tuning

model = Sequential()

#convolution 1st layer
model.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel), padding='same', activation='relu', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Dropout(droprate))

#convolution 2nd layer
model.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(droprate))

#convolution 3rd layer
model.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(droprate))

#Fully connected 1st layer
model.add(Flatten())
model.add(Dense(128,use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(droprate))

#Fully connected final layer
model.add(Dense(36))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=10, batch_size=512, verbose=1, validation_data=(X_val, Y_val), shuffle=True, callbacks=callbacks)

model.save('cnn_model_training1.h5')