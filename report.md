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

#### About the Dataset
Stanford has curated a ~600,000-image [dataset](http://ufldl.stanford.edu/housenumbers/) of house numbers from Google's [street view](https://www.google.com/streetview/).

Stanford's entire Street View House Numbers ([SVHN](http://ufldl.stanford.edu/housenumbers/)) dataset contains 4GB of data, or approximately 600,000 digit images.  These digit-images are provided either as **standalone 32x32 cropped images**, or as **larger images containing sequences** of individual digits in varying alignments and with varying spacing.

<img src="http://ufldl.stanford.edu/housenumbers/32x32eg.png" alt="32x32 cropped images">  
**32x32 Cropped Images**

<img src="http://ufldl.stanford.edu/housenumbers/examples_new.png" alt="larger multi-digit images with bounding boxse shown in blue" width=400>  
**Larger, Uncropped Images**

Also provided in `Matlab` format are "bounding boxes" which denote the location and label of individual digits within the image, for use in training the model.  Those boxes are shown in blue above for emphasis.


Teams of Machine Learning researchers have, for years, tried to correctly recognize the digits contained within these images, and have posted some [remarkable results](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#5356484e).



### Problem Statement
This project seeks to identify and output numbers which are contained in images.


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
