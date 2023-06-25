# Traffic Sign CNN

**Overview**
Artificial Intelligence fascinates me. To satisfy these interests, I have spent my time outside of my computer science curriculum teaching myself machine learning, deep learning, neural networks, and computer vision.

To develop my skills in deep learning and computer vision, I trained a Convolutional Neural Network to classify eight common traffic signs: Stop, Yield, Do Not Enter, No U-Turn, No Left Turn, No Right Turn, One Way (Left), and One Way (Right).

First, I custom-built a dataset by creating miniature, freestanding traffic signs, and took dozens of pictures of each sign. I then applied OpenCV transformations to each image to artificially enlarge the dataset.

I used the ResNet-50 Neural Network architecture to train the dataset. I chose this architecture experimentally, as it yielded the best accuracies of all the architectures that I tested over numerous runs.

During the initial stages of training, I only achieved accuracies that were at best 60-80% on the validation set. After experimenting with numerous architectures and tuning the hyperparameters, I achieved a 94.04% accuracy on the validation set and a 95.24% accuracy on the testing set.
**Instructions**
1) First, run *trainer.py*. This file will train the model on the dataset and create a pickle file for the test set. If the validation accuracy exceeds the accuracy value in *trainer.py*, it will save the model, which is critical for *predict.py*.

2) Second, run *trainer.py*. This file will load the trained model and display the images, the model's predictions for the images, and their actual values using Matplotlib.

**Visualization**
Here's a visualization of the model of the test set. Images with black text are correct while images with red text are incorrect:
<img width="1208" alt="Screenshot 2023-06-24 at 3 48 47 PM" src="https://github.com/spencerkarofsky/trafficSignCNN/assets/105813301/11792b22-c98b-4dcb-b407-6d3c7794fcb2">

Here's an expanded view of the model of the complete test set (168 images). The test set achieved 95.24% accuracy.
<img width="1391" alt="Screenshot 2023-06-23 at 10 00 33 PM" src="https://github.com/spencerkarofsky/trafficSignCNN/assets/105813301/8dd75d7e-2da3-455f-bfcc-378ab8963b9d">

Here's a visualization of the *augmentImage()* function. The top left image is the base image, while the rest are all augmented versions of the base image.
<img width="1456" alt="Screenshot 2023-06-14 at 9 56 24 PM" src="https://github.com/spencerkarofsky/trafficSignCNN/assets/105813301/78448ad5-c407-4759-a058-3b1b94421056">

Here's a graphical comparison of the performances of the different convolutional neural network architectures:
<img width="727" alt="Screenshot 2023-06-24 at 8 32 39 PM" src="https://github.com/spencerkarofsky/trafficSignCNN/assets/105813301/a33b96c2-d575-4566-affc-16298df68283">

