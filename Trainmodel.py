
#author mirror 
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD,RMSprop
from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

batch_size = 200
nb_classes = 2
nb_epoch = 20
train_label = np_utils.to_categorical(train_label, nb_classes)

#build the model 

model = Sequential()
model.add(Dense(600,input_shape = (15,)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(300))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model.fit(train_data,train_label,batch_size = batch_size, verbose=2, epochs=12, validation_split=0.2)
model.save("mymodel.h5")
