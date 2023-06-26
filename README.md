# Classifying Traffic Signs with Deep Learning and Convolutional Neural Networks

**Overview**

Artificial Intelligence is a professional interest of mine, and I have spent my time outside of my computer science curriculum teaching myself machine learning, deep learning, neural networks, and computer vision.

To develop my skills in deep learning and computer vision, I trained a Convolutional Neural Network to classify eight common traffic signs: Stop, Yield, Do Not Enter, No U-Turn, No Left Turn, No Right Turn, One Way (Left), and One Way (Right).

First, I custom-built a dataset by creating miniature, freestanding traffic signs, and took dozens of pictures of each sign. I then applied OpenCV transformations to each image to artificially enlarge the dataset.

<img width="1456" alt="augmentImage-visualization" src="https://github.com/spencerkarofsky/trafficSignCNN/assets/105813301/8e010f88-2ad2-41a5-be96-879228421533">
<sub>Here's a visualization of the *augmentImage()* function. The top left image is the base image, while the rest are all augmented versions of the base image.</sub>



I used the ResNet-50 Neural Network architecture to train the dataset. I chose this architecture experimentally, as it yielded the best accuracies of all the architectures that I tested over numerous runs.

<img width="727" alt="cnns-comparison" src="https://github.com/spencerkarofsky/trafficSignCNN/assets/105813301/5108f129-d575-4dd8-9b16-aa06dc5d0470">
<sub>Graphical comparison of the performances of the different convolutional neural network architectures</sub>


During the initial stages of training, I only achieved accuracies that were at best 60-80% on the validation set. After experimenting with numerous architectures and tuning the hyperparameters, I achieved a 94.04% accuracy on the validation set and a 95.24% accuracy on the testing set.

<img width="1208" alt="traffic-sign-cnn" src="https://github.com/spencerkarofsky/trafficSignCNN/assets/105813301/a530b37f-4e12-4d0f-b2fd-ac34fb302b0b">
<sub>Visualization of the model of the test set. Images with black text are correct while images with red text are incorrect:</sub>



<img width="1391" alt="traffic-sign-cnn-expanded" src="https://github.com/spencerkarofsky/trafficSignCNN/assets/105813301/3da02a7f-3d65-4be8-886e-bd9f3124d8de">
<sub>Expanded view of the model of the complete test set (168 images). The test set achieved 95.24% accuracy.</sub>



**Instructions**

1) First, run *trainer.py*. This file will train the model on the dataset and create a pickle file for the test set. If the validation accuracy exceeds the accuracy value in *trainer.py*, it will save the model, which is critical for *predict.py*.

2) Second, run *trainer.py*. This file will load the trained model and display the images, the model's predictions for the images, and their actual values using Matplotlib.
