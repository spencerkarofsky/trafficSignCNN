import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import keras_tuner as kt
import pickle

HEIGHT = 224
WIDTH = 224
'''
FUNCTIONS
'''
# augmentImages function enlarges the dataset by augmenting an image image to n number of images.
def augmentImage(img):
    # Rotation
    # Set random rotation angle that follows a normal distribution
    center = (WIDTH / 2, HEIGHT / 2)
    angle = (int)(np.random.normal(0,10)) # large standard deviation of 10 degrees so that the model is trained on a large variety of angles
    if angle < 0:
        angle = 360 + angle  # Convert negative angle to positive angle
    rot_matrix = cv2.getRotationMatrix2D(center,angle,1.0)
    img = cv2.warpAffine(img, rot_matrix,(WIDTH,HEIGHT))

    # Translation -- shift amounts follow normal distribution
    shift_x = np.random.normal(0,5) # ~68% of values will fall between -5 and 5
    shift_y = np.random.normal(0,5) # ~95% of values will fall between -10 and 10
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]]) # transformation matrix
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # Contrast Adjustment
    contrast_factor = np.random.normal(1.5,.5)
    img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)

    # Brightness Adjustment
    brightness_factor = np.random.normal(1.5,.4)
    img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

    return img

# Pulls the jpeg images to folders and creates a list of images
# Code sourced from:
#   https://www.geeksforgeeks.org/how-to-iterate-through-images-in-a-folder-python/#
#   https://www.geeksforgeeks.org/python-opencv-cv2-rotate-method/
# width and height of training images is 224*224 pixels
# https://www.youtube.com/watch?v=XrCAvs9AePM&ab_channel=LearnCodeByGaming
def imgFolderToList(path, list,n):
    for p in os.listdir(path):
        img = cv2.imread(os.path.join(path, p))
        img = cv2.resize(img, (WIDTH, HEIGHT))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        list.append(img)
        for i in range(n):
            aug_img = img
            aug_img = augmentImage(img)
            list.append(aug_img)
    if list is None:
        print('list empty')

# Convert jpeg image list to NumPy array
def list_to_arr(image_list):
    resized_images = []
    for img in image_list:
        resized_img = cv2.resize(img, (WIDTH, HEIGHT))
        resized_images.append(resized_img)
    arr = np.stack(resized_images)
    arr = arr.astype('float32') / 255.0
    return arr


'''
CHECK GPU SUPPORT

This step is important because it uses the GPU (Graphics Processing Unit) instead of the CPU (Central Processing Unit)
Using the GPU generally makes deep learning tasks run much faster.

Enabling GPU Support code sourced from: https://medium.com/mlearning-ai/install-tensorflow-on-mac-m1-m2-with-gpu-support-c404c6cfb580
'''
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

'''
PREPROCESSING
'''
# Define folder paths
stop_sign = './stop-sign/'
no_uturn = './no-uturn'
yield_path = './yield'
dne_path = './do-not-enter'
one_way_l_path = './1-way-left'
one_way_r_path = './1-way-right'
no_l_turn_path = './no-left-turn'
no_r_turn_path = './no-right-turn'

# Initialize lists
stops = []
nouturns = []
yields = []
dnes = []
onewayls = []
onewayrs = []
nolturns = []
norturns = []

AUGMENT_SIZE = 5 # Number of augment images created for each image in the dataset

# Add jpeg images from folders to list
imgFolderToList(stop_sign, stops,AUGMENT_SIZE)
imgFolderToList(no_uturn, nouturns,AUGMENT_SIZE)
imgFolderToList(yield_path,yields,AUGMENT_SIZE)
imgFolderToList(dne_path,dnes,AUGMENT_SIZE)
imgFolderToList(one_way_l_path,onewayls,AUGMENT_SIZE)
imgFolderToList(one_way_r_path,onewayrs,AUGMENT_SIZE)
imgFolderToList(no_l_turn_path,nolturns,AUGMENT_SIZE)
imgFolderToList(no_r_turn_path,norturns,AUGMENT_SIZE)

# Convert jpeg image lists to NumPy array lists and normalize values so that TensorFlow can process the data
stops = list_to_arr(stops)
nouturns = list_to_arr(nouturns)
yields = list_to_arr(yields)
dnes = list_to_arr(dnes)
onewayls = list_to_arr(onewayls)
onewayrs = list_to_arr(onewayrs)
nolturns = list_to_arr(nolturns)
norturns = list_to_arr(norturns)

# Create labels for each image array
stop_labels = np.full((len(stops),), 'stop sign')
nouturn_labels = np.full((len(nouturns),), 'no u-turn')
yield_labels = np.full((len(yields),), 'yield')
dne_labels = np.full((len(dnes),), 'do not enter')
one_way_l_labels = np.full((len(onewayls),), 'one way -- left')
one_way_r_labels = np.full((len(onewayrs),), 'one way -- right')
no_l_turn_labels = np.full((len(nolturns),), 'no left turn')
no_r_turn_labels = np.full((len(norturns),), 'no right turn')

# Concatenate the image array
images = np.concatenate((stops, nouturns, yields, dnes, onewayls, onewayrs, nolturns,norturns), axis=0) # X values

# Create labels by encoding the strings to integer values
label_dict = {
    'stop sign': 0,
    'no u-turn': 1,
    'yield': 2,
    'do not enter': 3,
    'one way -- left': 4,
    'one way -- right': 5,
    'no left turn': 6,
    'no right turn': 7
}
labels_encoded = np.array([label_dict[label] for label in np.concatenate((stop_labels, nouturn_labels, yield_labels,dne_labels, one_way_l_labels,one_way_r_labels,no_l_turn_labels,no_r_turn_labels))])

# Convert the encoded labels to one-hot encoded format
labels = tf.keras.utils.to_categorical(labels_encoded) # y values

# Randomize the orders of the arrays for training/testing
# Use same random state for images and labels so that the images and their corresponding labels are aligned
random_state = np.random.get_state()
np.random.shuffle(images)
np.random.set_state(random_state)
np.random.shuffle(labels)


# Divide into X_train, y_train, X_val, y_val for training and validation sets
# images are X; labels are y
TRAIN_SIZE = .8 # the percentage of the list that is used for training. Rounded to the nearest integer
VAL_SIZE = .15 # percentage of validation test set. rounded to the nearest integer

train_size = (int)((len(images) + .5) * TRAIN_SIZE)
val_size = (int)((len(images) + .5) * VAL_SIZE)


# Split data into training, validation, and test sets
X_train = images[:train_size]
y_train = labels[:train_size]
X_val = images[train_size:train_size+val_size]
y_val = labels[train_size:train_size+val_size]
X_test = images[train_size+val_size:]
y_test = labels[train_size+val_size:]

# Save test sets to pickle files to predict in predict.py
test_set = {
    'X_test': X_test,
    'y_test': y_test
}

with open("test_data.pkl", "wb") as f:
    pickle.dump(test_set, f)


print(f'Training data size: {len(X_train)}')
print(f'Validation data size: {len(X_val)}')
print(f'Test data size: {len(X_test)}')
print(f'\nTotal dataset size: {len(images)}')

'''
CONVOLUTIONAL NEURAL NETWORK
'''

# Create the ResNet152 model
cnn = tf.keras.applications.resnet.ResNet50(include_top=True, weights=None, input_shape=(WIDTH, HEIGHT, 3), pooling='max', classes=8)

# Compile the model
cnn.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Sourced from: https://towardsdatascience.com/a-practical-introduction-to-early-stopping-in-machine-learning-550ac88bc8fd
# Stop training cnn when val_accuracy achieves 90% accuracy
# Define the EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, mode='max', baseline=0.9, verbose=1)

# Fit the CNN to the training data
cnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = cnn.evaluate(X_val, y_val)

# Save the highest accuracy of all runs in a file
# If the current accuracy is greater than the accuracy in the file, save the cnn (new trained model is more accurate)
# Goal is to achieve 90% accuracy, so if accuracy is above 90%, print that accuracy goal has been achieved
with open('accuracy.txt', 'r') as f:
    historic_accuracy = float(f.readline())
# Save the trained model
if accuracy > historic_accuracy:
    with open('accuracy.txt', 'w') as f:
        f.write(str(accuracy))
    cnn.save('traffic-sign-cnn')
    if accuracy > .9:
        print(f'Model achieved an accuracy of {accuracy*100:.2f}%')
        print('This value exceeds the predetermined 90% accuracy threshold')




'''
# Graph Loss and Accuracy of the Model

# Plot images from X_test with the CNN's predictions and their actual values
num_images = (int)(len(X_val)/10)
cols = 4
rows = (num_images + cols - 1) // cols  # Calculate the number of rows based on the number of images

fig, axes = plt.subplots(rows, cols, figsize=(45, 45))  # Increase the figsize for larger images

for i, ax in enumerate(axes.flatten()):
    if i < num_images:
        img = X_val[i]  # Get the i-th image array
        ax.imshow(img)  # Plot the image
        ax.axis('off')
        y_pred = cnn.predict(img[np.newaxis, ...])  # Add additional dimension for batch
        predicted_class = np.argmax(y_pred)  # Get the index of the predicted class
        y_true_index = np.argmax(y_val[i])  # Get the index of the actual class
        y_true_label = [k for k, v in label_dict.items() if v == y_true_index][0]  # Retrieve the label from the label_dict
        predicted_label = [k for k, v in label_dict.items() if v == predicted_class][0]  # Retrieve the predicted label
        title = 'Prediction: ' + predicted_label + '\nActual: ' + y_true_label  # Combine the predicted and actual labels
        if not (predicted_label == y_true_label):
            ax.set_title(title,color='red')
        else:
            ax.set_title(title, color='black')
    else:
        ax.axis('off')  # Hide empty subplots

plt.tight_layout()  # Adjust the spacing between subplots
plt.show()'''