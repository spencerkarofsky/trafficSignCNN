import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import cv2
import numpy as np

# Load pickle file containing the test set
with open("test_data.pkl", "rb") as f:
    test_set = pickle.load(f)

X_test = test_set['X_test']
y_test = test_set['y_test']

# Dictionary of traffic sign labels
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

# Load the previously trained model from trainer.py
# Code sourced from: https://dontrepeatyourself.org/post/save-and-load-models-with-tensorflow/
cnn = tf.keras.models.load_model('traffic-sign-cnn')



'''
PLOTTING
'''
# Graph Loss and Accuracy of the Model

# Plot images from X_test with the CNN's predictions and their actual values
num_images = len(X_test)  # Number of images to plot (8 rows and 6 columns)
cols = 4  # Number of columns in the grid
rows = (num_images + cols - 1) // cols  # Calculate the number of rows based on the number of images

fig, axes = plt.subplots(rows, cols, figsize=(80, 80))  # Create a figure with subplots

for i, ax in enumerate(axes.flatten()):
    img = X_test[i]  # Get the i-th image array
    ax.imshow(img)  # Plot the image
    ax.axis('off')  # Turn off the axis

    # Get the CNN's prediction for the image
    y_pred = cnn.predict(np.expand_dims(img, axis=0))

    # Get the predicted and actual labels
    predicted_class = np.argmax(y_pred)
    predicted_label = [k for k, v in label_dict.items() if v == predicted_class][0]
    actual_label = [k for k, v in label_dict.items() if v == np.argmax(y_test[i])][0]

    # Combine the predicted and actual labels in the title
    title = 'Prediction: ' + predicted_label + '\nActual: ' + actual_label

    if predicted_label != actual_label:
        ax.set_title(title, color='red')
    else:
        ax.set_title(title, color='black')

plt.tight_layout()  # Adjust the spacing between subplots
plt.show()
