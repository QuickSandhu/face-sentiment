__author__ = 'rsandhu'

import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

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
data_augmentation=False
nb_epoch = 2000
batch_size = 15
valid_size = 20
valid_size_percent = 0.1

print("epochs: {}, batch-size: {}, validation-set-size: {} ".format(nb_epoch, batch_size, valid_size))



#################
#   Load Data   #
#################
from PIL import Image
import csv
img_names = os.listdir(img_dir)
img_names.sort()

sentiment_dict = {'AN' : 0,
                  'DI' : 1,
                  'SU' : 2,
                  'FE' : 3,
                  'HA' : 4,
                  'NE' : 5,
                  'SA' : 6}

# create an empty matrix of zeros that will hold input images for training
X = np.zeros((img_names.__len__(), CHANNELS, PIXELS, PIXELS), dtype='float32')
Y = np.zeros((img_names.__len__(),1), dtype='uint8')
i = 0
for filename in img_names:
    im = Image.open(img_dir + filename)
    X[i] = np.asarray(im, dtype='float32')
    sentiment = filename[3:5]
    Y[i,0] = sentiment_dict[sentiment]
    i += 1

X /= 255.0
Y = np_utils.to_categorical(Y)
Y_test, Y_train = Y[:valid_size,:], Y[valid_size:,:]
X_test, X_train = X[:valid_size,:], X[valid_size:,:]



#################
#   The Model   #
#################
# input: 256 x 256 images with 1 channel
# this applies 32 convolution filters of size 3 x 3 each

model = Sequential()
model.add(MaxPooling2D(pool_size=(2,2), input_shape=(1,256,256), border_mode='same'))

model.add(Convolution2D(32,3,3, border_mode='same', init='lecun_uniform'))
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32,3,3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))

model.add(Convolution2D(32,3,3, border_mode='same', init='lecun_uniform'))
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32,3,3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7))
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
    datagen.fit(X_train)

    hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                samples_per_epoch=X_train.shape[0],
                                nb_epoch=nb_epoch,
                                show_accuracy=True,
                                validation_data=(X_test, Y_test)
                                )
    print(hist.history)
else:
    print("training model...")
    hist = model.fit(X, Y,
                     batch_size = batch_size,
                     nb_epoch = nb_epoch,
                     validation_split=valid_size_percent,
                     show_accuracy=True,
                     shuffle=True, verbose=2
                     )
    print(hist.history)
    # loss, accuracy = model.evaluate(X, Y, show_accuracy=True, verbose=True)
