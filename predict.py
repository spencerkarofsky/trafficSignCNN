import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

with open("test_data.pkl", "rb") as f:
    test_set = pickle.load(f)

X_test = test_set['X_test']
y_test = test_set['y_test']

