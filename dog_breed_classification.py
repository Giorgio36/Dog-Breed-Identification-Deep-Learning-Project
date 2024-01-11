import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os import makedirs, listdir
import PIL
import numpy as np
import scipy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import shutil
import h5py

from keras.applications import ResNet50V2
from keras.applications import InceptionResNetV2

breed_df = pd.read_csv('D:/ML/IBM/Project/labels.csv')
train = 'D:/ML/IBM/Project/train'
test =  'D:/ML/IBM/Project/test'

breed_list = breed_df.breed.unique().tolist()
num_breeds = len(breed_list)

breed_df['file_name'] = breed_df['id'] + '.jpg'

# #Just run once
# dataset_home = 'D:/ML/IBM/Project/train'
# labeldirs = breed_list
# for labeldir in labeldirs:
#   newdir = os.path.join(dataset_home,labeldir)
#   makedirs(newdir,exist_ok = True)

# dirs = listdir('D:/ML/IBM/Project/train')
# s_folder = 'D:/ML/IBM/Project/train'
# d_folder = 'D:/ML/IBM/Project/train'
# for element in breed_df.index:
#   source = os.path.join(s_folder,str(breed_df.file_name[element]))
#   destination = os.path.join(d_folder,str(breed_df.breed[element]))
#   shutil.move(source,destination)
#   print('Moving elements...')

img_width = 224
img_height = 224
batch_size = 64

train_gen = ImageDataGenerator(validation_split = 0.2, rescale = 1.0/255.0, rotation_range= 45, zoom_range = 0.2, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip = True)
train_set = train_gen.flow_from_directory(directory = train, seed = 10, class_mode = 'sparse', batch_size = batch_size, shuffle = True, target_size = (img_width, img_height), subset = 'training')

val_gen = ImageDataGenerator(validation_split = 0.2, rescale = 1.0/255.0, rotation_range= 45, zoom_range = 0.2, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip = True)
val_set = train_gen.flow_from_directory(directory = train, seed = 10, class_mode = 'sparse', batch_size = batch_size, shuffle = True, target_size = (img_width, img_height), subset = 'validation')

test_gen = ImageDataGenerator(rescale = 1.0/255.0)
test_set = test_gen.flow_from_directory(directory = 'D:/ML/IBM/Project', classes = ['test'], batch_size = batch_size, target_size = (img_width, img_height))


#Checking for GPU------------------------------
device_name = tf.test.gpu_device_name()
if len(device_name) > 0:
    print("Found GPU at: {}".format(device_name))
else:
    device_name = "/device:CPU:0"
    print("No GPU, using {}.".format(device_name))


def build_compile_fit_adam(basemodel):
  # x = Flatten()(basemodel.output)
  x = GlobalAveragePooling2D()(basemodel.output)
  x = Dense(1024, activation = 'relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(512, activation = 'relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(num_breeds, activation = 'softmax')(x)

  model = Model(basemodel.input, x)

  learning_rate = 0.001
  optimizer = keras.optimizers.Adam(learning_rate = 0.001)
  with tf.device(device_name):
    model.compile( loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    # callbacks = [EarlyStopping(monitor = 'loss', patience = 5, mode = 'min', min_delta=0.01)]

    model.fit(train_set, validation_data = val_set, epochs = 20, steps_per_epoch = len(train_set), validation_steps = len(val_set), batch_size = batch_size)

  return model


def build_compile_fit_adadelta(basemodel):
  # x = Flatten()(basemodel.output)
  x = GlobalAveragePooling2D()(basemodel.output)
  x = Dense(1024, activation = 'relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(512, activation = 'relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(num_breeds, activation = 'softmax')(x)

  model = Model(basemodel.input, x)

  learning_rate = 0.001
  optimizer = keras.optimizers.Adadelta(learning_rate = 0.001)

  with tf.device(device_name):
    model.compile( loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    # callbacks = [EarlyStopping(monitor = 'loss', patience = 5, mode = 'min', min_delta=0.01)]

    model.fit(train_set, validation_data = val_set, epochs = 20, steps_per_epoch = len(train_set), validation_steps = len(val_set), batch_size = batch_size)

  return model


#Using transfer learning | ResNet50V2
# #Base model
# resnet50_v2 = ResNet50V2(include_top = False, input_shape = (img_width, img_height, 3), weights = 'imagenet')

# for layer in resnet50_v2.layers:
#   layer.trainable = False

# history_resnet50_v2 = build_compile_fit_adam(resnet50_v2)

# history_resnet50_v2.save_weights('./checkpoints/my_checkpoint_resnet50_v2')




#Using transfer learning | InceptionResV2

inceptionresv2 = InceptionResNetV2( include_top = False, input_shape = (img_width, img_height, 3), weights = 'imagenet')

for layer in inceptionresv2.layers:
  layer.trainable = False

history_inceptionresv2 = build_compile_fit_adam(inceptionresv2)

history_inceptionresv2.save('D:/ML/IBM/Project/inceptionresv2.h5')
