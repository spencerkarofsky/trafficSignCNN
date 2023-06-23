import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import cv2

# Load pickle file containing the test set
with open("test_data.pkl", "rb") as f:
    test_set = pickle.load(f)

X_test = test_set['X_test']
y_test = test_set['y_test']

print(X_test[4].shape)
print(y_test[4].shape)
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

# Reshape the input data to match the expected shape
X_test_reshaped = X_test.reshape(-1, 224, 224, 3)

# Run the first 16 elements of the test set through the CNN
for i in range(16):
    prediction = cnn.predict(X_test_reshaped[i])
    predicted_label = label_dict[prediction.argmax()]
    actual_label = label_dict[y_test[i]]
    print("Predicted label:", predicted_label)
    print("Actual label:", actual_label)
    print()