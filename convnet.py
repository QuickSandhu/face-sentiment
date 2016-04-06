__author__ = 'rsandhu'

import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

##############################
#  Machine Dependendent Vars #
##############################
img_dir = '/Users/sraymon/face-sentiment/jaffe/'
ratings_path = '/Users/sraymon/face-sentiment/sentiment-ratings-sorted.csv'

###########
#  Vars   #
###########
PIXELS = 256
CHANNELS = 1
data_augmentation=True
nb_epoch = 10
batch_size = 16
valid_size = 21

#################
#   Load Data   #
#################
from PIL import Image
import csv
img_names = os.listdir(img_dir)
img_names.sort()

Y = []
with open(ratings_path, 'r') as file:
    reader = csv.reader(file, delimiter=' ')
    for rowdata in reader:
        rowdata = [float(rowdata[1]), float(rowdata[2]), float(rowdata[3]), float(rowdata[4]),float(rowdata[5]), float(rowdata[6])]
        Y.append(rowdata)

Y = np.asarray(Y) / 5 # Jaffe images rated out of 5
# convert to probabilities
i = 0
for row in Y:
    Y[i,] = Y[i,] / sum(Y[i,])
    i += 1

# create an empty matrix of zeros that will hold input images for training
X = np.zeros((img_names.__len__(), CHANNELS, PIXELS, PIXELS), dtype='float32')
i = 0
for filename in img_names:
    im = Image.open(img_dir + filename)
    X[i] = np.asarray(im, dtype='float32') / 255
    i += 1

Y_test, Y_train = Y[:valid_size,:], Y[valid_size:,:]
X_test, X_train = X[:valid_size,:], X[valid_size:,:]

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

model.add(Convolution2D(32,3,3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3, border_mode='valid'))
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
if data_augmentation:
    print('Using data augmentation')
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)
    datagen.fit(X)

    hist = model.fit_generator(datagen.flow(X, Y, batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        show_accuracy=True,
                        validation_data=(X_test, Y_test)
                        )
    print(hist.history)
else:
    print("training model...")
    hist = model.fit(X, Y,
                     batch_size = 10,
                     nb_epoch = 1,
                     show_accuracy=True,
                     )
    print(hist.history)
    loss, accuracy = model.evaluate(X, Y, show_accuracy=True, verbose=True)
