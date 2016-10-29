## Deep Learning Capstone Project
### Background
[Neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network) are effective [function approximators](http://neuralnetworksanddeeplearning.com/chap4.html), but it turns out that the deeper the neural net is, the [more complicated](https://en.wikipedia.org/wiki/Deep_learning#Applications) are the tasks it can perform.

Among these complicated tasks is digit recognition
![digit recognition](http://techglam.com/wp-content/uploads/2013/10/reCAPTCHA.jpg)

### Problem Statement
This project seeks to identify and output numbers which are contained in images.

### Dataset and Inputs
The MNIST and notMNIST datasets will be used to develop the neural network model.

Once a neural net model is in place, the SVHN dataset will be used to train the model.

Once this is done, the model will be fed images from the wild to see how it performs.

### Solution Statement
This will be accomplished through the creation and training of a deep neural net to recognize numeric content within an image.

### Benchmark Model
For comparison, [Goodfellow et al.](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf) achieved 97.84% per-digit accuracy, and this project attempts to approximate, but not achieve, that performance.

### Evaluation Metrics
Performance will be evaluated on a per-digit-recognition basis, with a target accuracy of 90% or better.  

### Project Design
