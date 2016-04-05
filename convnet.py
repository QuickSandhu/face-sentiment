__author__ = 'rsandhu'

import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD



#################
#   Load Data   #
#################
from PIL import Image
import csv
PIXELS = 256
CHANNELS = 1
img_dir = '/home/rsandhu/face-sentiment/face-sentiment/jaffe/'
img_names = os.listdir(img_dir)
img_names.sort()

Y = []
with open('/home/rsandhu/face-sentiment/face-sentiment/sentiment-ratings-sorted.csv', 'r') as file:
    reader = csv.reader(file, delimiter=' ')
    for rowdata in reader:
        rowdata = [float(rowdata[1]), float(rowdata[2]), float(rowdata[3]), float(rowdata[4]),float(rowdata[5]), float(rowdata[6])]
        Y.append(rowdata)

Y = np.asarray(Y) / 5 # Jaffe images rated out of 5


# create an empty matrix of zeros that will hold input images for training
X = np.zeros((img_names.__len__(), CHANNELS, PIXELS, PIXELS), dtype='float32')
i = 0
for filename in img_names:
    im = Image.open(img_dir + filename)
    X[i] = np.asarray(im, dtype='float32') / 255
    i += 1


#################
#   The Model   #
#################
# input: 256 x 256 images with 1 channel
# this applies 32 convolution filters of size 3 x 3 each
model = Sequential()
model.add(Convolution2D(32,3,3, border_mode='valid', input_shape=(1,256,256)))
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(6))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)



#################
#   Training    #
#################
print("training model...")
hist = model.fit(X, Y,
          batch_size = 10,
          nb_epoch = 200,
          )
print(hist.history)
loss, accuracy = model.evaluate(X, Y, show_accuracy=True, verbose=True)
