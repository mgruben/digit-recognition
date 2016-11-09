# Deep Learning Capstone Project Report
## Definition
### Project Overview
#### Background
While computers can easily digest and process text input like the sentence you're reading right now, the task of recognizing text that is depicted by an image has [historically](https://en.wikipedia.org/wiki/Timeline_of_optical_character_recognition) been much more difficult.  
![digit recognition](http://techglam.com/wp-content/uploads/2013/10/reCAPTCHA.jpg)

Early attempts relied on [heavily](https://en.wikipedia.org/wiki/OCR-A)-[standardized](https://en.wikipedia.org/wiki/OCR-B) fonts, responsible in large part for the iconic blocky numbers at the bottom of checks (7 below).  
<img src="https://upload.wikimedia.org/wikipedia/commons/8/8e/BritishChequeAnnotated.png" alt="An example check" width="400">

Understandably, this is not an ideal solution, since we would like to be able to feed a computer the exact same input that we ourselves receive.  
For example, ideally a computer would be able to tell us that the following image represents "31"  
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/House-number-21-cottage-Poland.jpg/320px-House-number-21-cottage-Poland.jpg" alt="An example street address" width=400>

**Digit recognition** is a subset of text recognition (a 10-character subset, to be exact), and is the focus of this project.

#### A Relevant Dataset
Stanford, in conjunction with Google, has curated a ~600,000-image [dataset](http://ufldl.stanford.edu/housenumbers/) of house numbers from Google's [street view](https://www.google.com/streetview/).

Stanford's entire Street View House Numbers ([SVHN](http://ufldl.stanford.edu/housenumbers/)) dataset contains 4GB of data, or approximately 600,000 digit images.  These digit-images are provided either as **standalone 32x32 cropped images**, or as **larger images containing sequences** of individual digits in varying alignments and with varying spacing.

<img src="http://ufldl.stanford.edu/housenumbers/32x32eg.png" alt="32x32 cropped images">  
**32x32 Cropped Images**

<img src="http://ufldl.stanford.edu/housenumbers/examples_new.png" alt="larger multi-digit images with bounding boxse shown in blue" width=400>  
**Larger, Uncropped Images**

Also provided in `Matlab` format are "bounding boxes" which denote the location and label of individual digits within the image, for use in training the model.  Those boxes are shown in blue above for emphasis.


### Problem Statement
#### Background
Numerous Machine Learning teams have [created models](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#5356484e) to recognize the digits contained in the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset.

There are two main ways to quantify their performance:  

1. **Whole-sequence Accuracy**  
   Of the [SVHN](http://ufldl.stanford.edu/housenumbers/) images, how many were correctly recognized without any flaws?

   This quantification does not award "partial credit," since 2134 Sycamore Lane is not the same as 1234 Sycamore Lane.

2. **Per-digit Accuracy**  
   Of the digits within the [SVHN](http://ufldl.stanford.edu/housenumbers/) images, how many were correctly recognized?  
   
   This quantification does award "partial credit," since we admit that 2134 Sycamore Lane is closer to correctly representing 1234 Sycamore Lane than is 851 Sycamore Lane.

One model in particular, developed by [Goodfellow et al.](https://arxiv.org/pdf/1312.6082v4.pdf) in 2014, achieved 96% whole-sequence accuracy, and just under 98% per-digit accuracy.

#### Statement
> This project seeks to identify and output digits which are contained in the larger, uncropped images from the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset with at least 96% whole-sequence accuracy.

#### Approach
In short, this project will create and train a [TensorFlow](https://www.tensorflow.org) deep neural net to recognize and output digits from [SVHN](http://ufldl.stanford.edu/housenumbers/) images.

In detail, the approach this project will take is as follows:

1. **Design and test** a model architecture that can identify sequences of digits in an image.
 1. This will largely follow the work of [Goodfellow et al.](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf), as they have already developed an effective and efficient model for this task.
 2. This project will use a deep neural network as implemented by the [TensorFlow](https://www.tensorflow.org) library.  "Deep" here refers to the fact that there are several hidden layers in the neural network.
 3. Model development will largely focus on the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset, likely by training with a [convolutional neural network](http://deeplearning.net/tutorial/lenet.html) in order to reduce the need for explicit image pre-processing.
 
    It is expected that the neural network will employ **softmax regressions** in order to choose between competing interpretations of a given digit image.
    
    For training on the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset, a subset of the already-provided training data will be used.

2. **Train** a model on realistic data.
 1. This phase will attempt to replicate the performance achieved on that dataset by [Goodfellow et al.](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf) on the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset, while acknowledging that their model will likely outperform mine.
 2. As suggested by [Goodfellow et al.](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf) (see their Figure 4 below), it is expected that additional model features, such as **specialized units** may be necessary in order to detect digits within the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset with sufficent accuracy.
 ![Accuracy vs. Depth](http://i.imgur.com/qItcORO.png)

3. Feed the model new number-containing **images from the wild**.

   This phase will involve one or both of the following:
  1. hand-photographing digits available locally, or
  2. Creating (e.g. drawing) digits, either [on-screen](https://www.youtube.com/watch?v=ocB8uDYXtt0) or on paper.
 
   After obtaining images from the wild, these images will be processed so that they are in a form which the neural net expects, and they will be input to the neural net to examine its digit-recognition performance.

4. **Localization** will be employed to display a box around detected sequences of digits.

   This will be made possible by meta-data within the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset, and as [Goodfellow et al.](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf) suggest, will likely require additional hidden layers to perform the localization task.  
   <img src="http://i.imgur.com/EX5it8P.png" alt="Localization" width="400">


### Metrics


## Analysis
### Data Exploration
### Exploratory Visualization
### Algorithms and Techniques
### Benchmark

## Methodology
### Data Preprocessing
### Implementation
### Refinement

## Results
### Model Evaluation and Validation
### Justification

## Conclusion
### Free-Form Visualization
### Reflection
### Improvement
