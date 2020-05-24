import numpy as np
import tensorflow as tf
import glob
import os
from os import walk
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
import pathlib
import sklearn as sk
import cv2
from sklearn.model_selection import train_test_split

# ssh openhack@35.202.246.73


if __name__ == '__main__':

    DATADIR = "covid-19-x-ray-10000-images/dataset"
    CATAGORIES = ['Covid', 'Normal']

    for cat in CATAGORIES:
        path = os.path.join(DATADIR, cat) #path to cats or dogs
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            plt.imshow(img_arr, cmap="gray")
            plt.show()
            break
        break

    IMG_SIZE = 50

    new_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE ))

    training_data = []

    def create_train_data():
        for cat in CATAGORIES:
            path = os.path.join(DATADIR, cat)  # path to cats or dogs
            class_num = CATAGORIES.index(cat)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_arr, class_num])
                except Exception as e:
                    print("Error Image")
        return training_data

training_data1 = create_train_data()

print(len(training_data1))

import random

random.shuffle(training_data)

for sample in training_data:
    print(sample[1])

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
""""#reading in images
    images = np.array()
    #image_lables = []

    #arr = os.listdir("covid-19-x-ray-10000-images//dataset//covid")
    data_dir = pathlib.Path("covid-19-x-ray-10000-images//dataset")
    img_list_glob_covid = glob.glob("./covid-19-x-ray-10000-images//dataset//covid//*.jpeg")
    img_list_glob_normal = glob.glob("./covid-19-x-ray-10000-images//dataset//normal//*.jpeg")
    img_list = list(data_dir.glob('*/*.jpeg'))
    image_count = len(img_list)
    #print(img_list_glo)
    for file in img_list_glob_covid:
        #img = Image.open(file)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.read_file(file)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)
        # resize the image to the desired size.
        img = tf.image.resize(img, [800, 800])

        images.append(img)
        image_lables.append(1)
    for file in img_list_glob_normal:
        #img = Image.open(file)
        img = tf.read_file(file)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
       # img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        img = tf.image.resize(img, [800, 800])

        images.append(img)
        image_lables.append(0)


    data = np.array()
    #pixels = []
    #for im in images:
        #pixels.append(np.array(pic))
    #data = list(zip(pixels, image_lables))

    #Create train/test split
    train, test = train_test_split(im, test_size=0.33)

    x_train = train[:,1]
    x_test = train[:,1]
    y_train = train[:,1]
    y_test = train[:,1]


    #Create tensor data objects
    #images_train = np.expand_dims(train[:0], axis=1)
    #images_train = tf.cast(train[:0], tf.float32)
    #labels_train = tf.cast(train[:1], tf.float32)
    #dataset_train = tf.data.Dataset.from_tensor_slices((images_train, labels_train))



    #Model
    num_kernels = 64
    dense_layer_neurons = 1024
    kernels_size = (3, 3)
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(num_kernels, kernels_size, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),

        tf.keras.layers.Conv2D(num_kernels, kernels_size, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(dense_layer_neurons, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # Do not change any arguments in the call to model.compile()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )

    #model.fit(train, steps_per_epoch=5)

    #model.evaluate(x_test, verbose=2)

    print()"""
