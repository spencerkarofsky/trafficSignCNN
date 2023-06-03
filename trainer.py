import cv2
import os
import tensorflow as tf
import tensorflow.python.keras.layers as layers
import numpy as np

#define folder paths
stop_sign = '../cascades/stop-sign/'
negative_path = '../cascades/negative'

#define list for positive and negative images
stops = []
negatives = []

# read through positve image folder first
# rotate 90 degrees to proper orientation and resize to 28*28 pixel array
# store in positives[]
# Code sourced from:
#   https://www.geeksforgeeks.org/how-to-iterate-through-images-in-a-folder-python/#
#   https://www.geeksforgeeks.org/python-opencv-cv2-rotate-method/
#width and height of training images is 28*28 pixels
# https://www.youtube.com/watch?v=XrCAvs9AePM&ab_channel=LearnCodeByGaming
def imgFolderToList(path,list):
    width = 64
    height = 64
    for p in os.listdir(path):
        img = cv2.imread(os.path.join(path, p))
        img = cv2.resize(img,(width,height))
        list.append(img)


imgFolderToList(negative_path, negatives)
imgFolderToList(stop_sign,stops)

# Convert images to tensors (so Tensorflow can read the images)
# Images to Tensors code sourced from: https://www.binarystudy.com/2022/01/how-to-convert-image-to-tensor-in-tensorflow.html
neg_tensors = []
pos_tensors = []
for n in negatives:
    neg_tensors.append(tf.convert_to_tensor(n))
for p in stops:
    pos_tensors.append(tf.convert_to_tensor(p))

# Convert to NumPy arrays of image tensors
neg_array = np.array(neg_tensors)
pos_array = np.array(pos_tensors)

# Normalize values between 0 and 1
pos_array = pos_array.astype('float32') / 255.0
neg_array = neg_array.astype('float32') / 255.0

# Create labels for the positive and negative images.
# Positive images, where there is a stop sign, get a value of 1.
# negative images, where there is no stop sign, get a value of 0.
pos_labels = np.ones(len(pos_array))
neg_labels = np.zeros(len(neg_array))

# Combine positive and negative images and positive and negative labels
images = np.concatenate((pos_array,neg_array),axis=0)
labels = np.concatenate((pos_labels,neg_labels),axis=0)

# Convert labels to one-hot encoded format
labels = tf.keras.utils.to_categorical(labels)

# Randomize the order of the lists. Because there's 2 separate lists, need to ensure that both lists are randomized in the same order so they align
rand_state = np.random.get_state()
np.random.shuffle(images)
np.random.set_state(rand_state)
np.random.shuffle(labels)

# Divide into X_train, y_train, X_test, y_test for training and test sets
# images are X; labels are y
TRAIN_SIZE = .7 # the percentage of the list that is used for training. 1-TRAIN_SIZE = testing size. Train_size is rounded to nearest integer

train_size = (int) ((len(images)+.5) / TRAIN_SIZE)

X_train = images[:train_size]
y_train = labels[:train_size]
X_test = images[train_size:]
y_test = labels[train_size:]

# Define CNN sequential model
cnn = tf.keras.Sequential()

# Set up CNN architecture
cnn.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
cnn.add(layers.MaxPooling2D((2,2)))

cnn.add(layers.Dropout(rate=.3))

cnn.add(layers.Flatten())

cnn.add(layers.Dense(64,activation='relu'))
cnn.add(layers.Dense(2,activation='softmax'))

cnn.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])

# Run data through the CNN
cnn.fit(X_train,y_train,epochs=7,batch_size=32,validation_data=(X_test,y_test))

cnn.save('stop_sign_cnn')
