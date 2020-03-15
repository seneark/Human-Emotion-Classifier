import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


train_data_file = "./images/train"
test_data_file = "./images/validation"


#-------------------------Neural Network Layers--------------------------#

classifier = Sequential()

# Convolution and Pooling Layer 1
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), kernel_initializer='he_normal',
                             input_shape=(48, 48, 1), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.2))

# Convolution and Pooling Layer 2
classifier.add(Convolution2D(filters=64, kernel_size=(3, 3),
                             kernel_initializer='he_normal', activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.2))

# Convoltion and Pooling Layer 3
classifier.add(Convolution2D(filters=128, kernel_size=(3, 3),
                             kernel_initializer='he_normal', activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.2))

# Convoltion and Pooling Layer 4
classifier.add(Convolution2D(filters=256, kernel_size=(3, 3),
                             kernel_initializer='he_normal', activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.2))

# Flatten Layer
classifier.add(Flatten())

# Hidden Layers
classifier.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.4))

classifier.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.4))

# Output layer
classifier.add(Dense(5, kernel_initializer="he_normal", activation='softmax'))

# Compiling the classifier
classifier.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

#----------------------Images Reader-----------------------------#
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=30,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.4,
                                   height_shift_range=0.4,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('./images/train',
                                                 color_mode='grayscale',
                                                 target_size=(48, 48),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

test_set = test_datagen.flow_from_directory('./images/validation',
                                            color_mode='grayscale',
                                            target_size=(48, 48),
                                            batch_size=32,
                                            class_mode='categorical')


# Seeing the Summary
print(classifier.summary())


#------------------Creating a Model For classifier--------------------#
mdlCheckpoint = ModelCheckpoint(
    'Emotion_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

erlyStop = EarlyStopping(monitor='val_loss', min_delta=0,
                         patience=9, verbose=1, restore_best_weights=True)

reducePla = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)

callback = [erlyStop, mdlCheckpoint, reducePla]


#---------------------Fitting the Data--------------------#
train_sample = 24282
test_sample = 5864

history = classifier.fit_generator(training_set, steps_per_epoch=train_sample//32,
                                   epochs=50,
                                   callbacks=callback,
                                   validation_data=test_set,
                                   validation_steps=test_sample//32)
