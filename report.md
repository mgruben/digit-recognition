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
   <img src="images/localization.png" alt="Localization" width="400">


### Metrics
As mentioned above, two of the most common ways to measure performance on the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset are **whole-sequence accuracy** and **per-digit accuracy**.

This project will measure its performance through **whole-sequence accuracy**.

**Per-digit accuracy** will also be reported to allow for more-robust comparison between existing models, but this metric will not be the focus of this project.

## Analysis
![Sample Collection](images/Sample.png)
A random sample of images from the SVHN training dataset.

### Data Exploration  
#### Label Distribution  
Recall that the labels the SVHN dataset presents us with are numbers in the inclusive range (1, 10), where the labels 1 through 9 represent the digits 1 through 9, and the label 10 represents the digit 0.

So, what we see is fascinating: The digits are not uniformly distributed.  
![Training Set Label Frequency](images/TrainingSetLabelFrequency.png)  
We would expect a uniform distribution on rolls of a single die, or on the rank of cards drawn from a deck with replacement.  That is to say, we would expect uniformity if it were equally likely that any enumerated outcome could occur.

What we notice here, however, is that `1` is much more likely to occur in this dataset than is `0`, and that `0` is marginally more likely to occur in this dataset than is `9`, which is the least likely of all.  
Let's try to give more quantitative heft to this via **norming** our histogram:  
![Training Set Label Probability](images/TrainingSetLabelProbability.png)  
Revisiting our narrative from above, if we were to pick a digit at random from our dataset, there's ~20% chance that digit is a `1`, and a ~7.5% chance that digit is a `9`.

This is important to note, since if we see our model is correctly classify classifying 20% of the data, it could be the case that our model is just guessing `1` for every classification.

#### Image Cardinality
What is the distribution of the number of digits per image in the SVHN dataset?  
![Training Set Digit-Count Frequency](images/TrainingSetDigit-CountFrequency.png)  
Note that, in the above histogram, there are no images with `0` digits in them, and there is only one image with `6` digits in it.

It is thus perhaps the case that a Tensor which has been built to identify digits in a given image may be pre-disposed to look for two digits, regardless of whether there are 1 or 6 actually in the image.

### Exploratory Visualization
House numbers aren't always straight.  This is (1) because such precision is hard, and (2) because sometimes sloping numbers look better.

Let's explore how the numbers slope in the SVHN dataset.  
![Training Set Average Digit Angle Frequency, Linear Scale](images/TrainingSetAverageDigitAngleFrequencyLinear.png)  
Interestingly, as can be seen above, there are 9 images in which the slope of the numbers suggests a bottom-to-top ordering.  This almost certainly wasn't intentional.

Is this an error in the dataset?  If so, it affects only a few of the images.

Note also that there aren't any images in which the house numbers are angled beyond -90 degrees.  This suggests that no one placed numbers close enough to bottom-to-top orientation for the error inherent in placing numbers to cause that particular placement to exceed -90 degrees.  Neat!

Let's view the same plot, but with a logarithmically-scaled y-axis, to better see the above-identified outliers.  
![Training Set Average Digit Angle Frequency, Logarithmic Scale](images/TrainingSetAverageDigitAngleFrequencyLogarithmic.png)  
As we can see in the above histograms, house numbers are predominately flattish, but some slope upwards (have a positive `Angle`), and some slope downwards (have a negative `Angle`).

As expected, the tail on the negative side is fatter than on the positive side, meaning that more house numbers slope downwards than slope upwards.

This makes sense, as bottom-to-top ordering is not the reading convention anywhere.
![Writing Directions of the World](images/Writing_directions_of_the_world.svg)
(image [source](https://en.wikipedia.org/wiki/Writing_system#/media/File:Writing_directions_of_the_world.svg))  

#### Relevance to Analysis
This suggests that the house numbers are, by overwhelming majority, ordered left-to-right.  Thus, after having identified digits in the SVHN images, digits can be ordered in the output based off of their `x-coordinate`, or `training_metadata['left']` value as referred to above.

### Algorithms and Techniques
In short, this project will need to:  
1. Take images and their metadata as input,  
2. Preprocess the images according to that metadata,  
3. Define a neural net for learning from the preprocessed images.  
4. Train the neural net on the preprocessed images.  

This approach has its origins in Yann LeCun's [LeNet](https://www.youtube.com/watch?v=FwFduRA_L6Q) ([LeCun et al., 1995, Learning Algorithms for Classification: A Comparison on Handwritten Digit Recognition](http://s3.amazonaws.com/academia.edu.documents/30766359/10.1.1.41.6835.pdf?AWSAccessKeyId=AKIAJ56TQJRTWSMTNPEA&Expires=1483595347&Signature=x8un0ZGLkpik1ypX4HpGfPoX%2FSA%3D&response-content-disposition=inline%3B%20filename%3DLearning_algorithms_for_classification_A.pdf)), but has more recently been implemented to [recognize house numbers](https://www.youtube.com/watch?v=vGPI_JvLoN0) in Google Street View images (See [Goodfellow et al., 2013, Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](https://arxiv.org/pdf/1312.6082v4.pdf), the impetus for this project).

It is the goal of this project to do as little preprocessing to the images as possible.  
This is partly a desire to create an autonomous digit recognition platform, balanced against the desire to somehow reduce the size of the input (and thus lower the computational cost), balanced also against the `Keras` requirement that inputs to neural nets all have the same dimension.

The neural net implementation is, broadly speaking, a deep convolutional neural net, so-called because it will have several layers between the input and the output layers ("hidden" layers), some of which perform [convolutions](http://cs231n.github.io/convolutional-networks/) on the inputs.

For more details about the algorithms and techniques employed during [preprocessing](#image-preprocessing) and [model implementation](#convolutional-neural-net), please see those sections respectively.

### Benchmark  
The benchmark for this project is that achieved by Goodfellow et al. on the public SVHN dataset:  
96% whole-sequence accuracy.
## Methodology
### Data Preprocessing
The images in the SVHN dataset are many different sizes:
```
$ file *.png
...
28668.png:             PNG image data, 153 x 73, 8-bit/color RGB, non-interlaced
28669.png:             PNG image data, 67 x 34, 8-bit/color RGB, non-interlaced
2866.png:              PNG image data, 44 x 21, 8-bit/color RGB, non-interlaced
28670.png:             PNG image data, 100 x 50, 8-bit/color RGB, non-interlaced
28671.png:             PNG image data, 83 x 34, 8-bit/color RGB, non-interlaced
28672.png:             PNG image data, 108 x 49, 8-bit/color RGB, non-interlaced
...
```
Unfortunately, our neural net will need its input to be consistently-sized.  
That is to say, we need to pick *a* size to which we resize all images, before we feed them to the neural net.

Images which are larger than the destination size are going to *lose* some information, while images which are the destination size or smaller aren't going to gain information.  
Accordingly, we want to pick a size where information loss isn't significant.

A note here:  
Just because we downsize the image doesn't mean that there is significant information loss.  
A 3200x4800 image of the letter `A` is probably still faithfully represented if downsized to 32x48, or smaller.

Let's get an idea of the input dimensions:
![Distribution of Image Sizes](images/DistributionOfImageSizes.png)  
Notice that the above plot suggests that the vast majority of the images are less than 50-or-so pixels tall, and many of them are less than 100 pixels wide.

That is, we can probably downsize these images to 54 x 54 and (hopefully) not lose a lot of information.
![Distribution of Image Sizes Histogram](images/DistributionOfImageSizesHist.png)  
Let's go with 64x64, the size used by Goodfellow et al.  
But hold on, we can't just rescale our input images, or we'd get too-squished input like this:  
![Squished Input](images/squished.png)  
We need to be more clever about how we downsize.
#### Clever Image Preprocessing
[Goodfellow et al.](https://arxiv.org/pdf/1312.6082v4.pdf) describe the following steps to their image preprocessing:  

1. **Find the rectangular bounding box** that will contain individual character bounding boxes.  
   Note that this means finding `y_min`, `y_max`, `x_min`, and `x_max` such that, for each image, no bounding box is clipped.  
   Note further that, for each individual character bounding box, `x_min` is simply `training_metadata['left'][i]`, and `y_min` is `training_metadata['top'][i]` (the y-axis is positive in the downward direction).  
   Then, `y_max` can be found by adding `training_metadata['height'][i]` to `y_min`, and `x_max` can be found by adding `training_metadata['width'][i]` to `x_min`.  
   Then, we simply take the lowest of the `x_min` and `y_min`, and the largest of the `x_max` and `y_max`.

2. **Expand the rectangular bounding box** by 30% in the `x` and `y` directions.  
   This seems like they are intending to increase from the rectangle's centroid, so that if we have a `width` of 20, then `x_min` decreases by 3, and `x_max` increases by 3.
   
3.  **Crop the image** to this bounding box.  
   Note that, if the expanded bounding box extends beyond the image dimensions, the box will have to be cropped to fit the image.

4.  **Resize the image** to 64 x 64.  
   This step can make use of the pipeline we defined earlier, now that we have a better idea, for each image, where the pixels containing the digit information are located.
   
5.  **Crop several 54 x 54 images** from random locations within the 64 x 64 image.
   This step increases the size of the dataset, which is good for training purposes.
   Note that this step **may** cause us to lose some digit information:  
   1. For each image, we've defined a **rectangle** which bounds all digit boxes.  Therefore the digits are probably in that rectangle.  
   2. Then, we upsized the box by 30%, which is a 15% increase on each axis, on each side.  
   3. Then we resize, which decreases resolution, but doesn't change the relative location of boundaries.  
   4. *Then*, we crop from to 54 pixels from 64 pixels.  
   Since we would have to multiply 54 by 18.52% to get back to 64, if the pixels were all cropped from one side (as might happen due to randomness), then we may sometimes crop **inside** of the original digit-containing rectangle, and maybe even cut into some digits.  
   
   This is probably not a huge deal, but it's worth noting.
   
6. **Subtract the mean** of each image.

### Implementation
#### Image Preprocessing
Let's implement this preprocessing pipeline on the following sample image:
```python
i = 31750
```
![Pipeline Input](images/PipelineInput.png)  
##### Find the Rectangular Bounding Box
```python
def getBBox(i, train=True):
    '''
    Given i, the desired i.png, returns
    x_min, y_min, x_max, y_max,
    the four numbers which define the small rectangular bounding
    box that contains all individual character bounding boxes
    '''
    if train:
        metadata = training_metadata
    else:
        metadata = testing_metadata
    
    x_min = min(metadata['left'][i-1])
    y_min = min(metadata['top'][i-1])
    x_max = max(map(add, metadata['left'][i-1], metadata['width'][i-1]))
    y_max = max(map(add, metadata['top'][i-1], metadata['height'][i-1]))
    return x_min, y_min, x_max, y_max
```
![Find the Rectangular Bounding Box](images/PipelineFoundBox.png)  
##### Expand the Rectangular Bounding Box
```python
def expandBBox(x_min, y_min, x_max, y_max):
    '''
    Given the four boundaries of the bounding box, returns
    those boundaries expanded out from the centroid by 30%, as
    x_min, y_min, x_max, y_max
    '''
    # The delta will be 30% of the width or height, (integer) halved
    x_d = ((x_max - x_min) * 0.3) // 2
    y_d = ((y_max - y_min) * 0.3) // 2
    return x_min - x_d, y_min - y_d, x_max + x_d, y_max + y_d
```
![Expand the Rectangular Bounding Box](images/PipelineExpandedBox.png)  
##### Crop the Image
```python
def cropBBox(img, x_min, y_min, x_max, y_max):
    '''
    Given a numpy array representing an image, and
    the four boundaries of the bounding box, returns
    the cropped bounding box, as
    x_min, y_min, x_max, y_max
    '''
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img.shape[1], x_max)
    y_max = min(img.shape[0], y_max)
    return x_min, y_min, x_max, y_max
```
![Crop the Image](images/PipelineCropped.png)
##### Resize the Image
```python
def getResized(f, train=True):
    '''
    Given an open file f, representing the desired image file,
    and a boolean representing whether this is a training image
    or a testing image,
    returns a numpy array, which is the portion of the
    image enclosed by the bounding box around all digits,
    resized to:
    64 pixels by 64 pixels if train=True,
    54 pixels by 54 pixels if train=False
    '''
    # Read the file as a numpy array
    img = mpimg.imread(f)
    
    # Get the index i from our filename
    if train:
        i = int(f.name[11:].split('.')[0])
    else:
        i = int(f.name[10:].split('.')[0])

    # Get our final expanded, cropped digit-bounding box
    x_min, y_min, x_max, y_max = cropBBox(img, *expandBBox(*getBBox(i, train)))
    
    # Return the cropped, resized numpy array
    if train:
        return misc.imresize(img[y_min:y_max, x_min:x_max], (64,64))
    else:
        return misc.imresize(img[y_min:y_max, x_min:x_max], (54,54))
```
![Resize the Image](images/PipelineResized.png)  
##### Randomly Crop Several Smaller Images
```python
def getRandomSmaller(img):
    '''
    Given img, a 64 x 64 numpy array representing an image,
    returns a randomly-sliced 54 x 54 numpy array,
    representing a crop of the original image
    '''
    x_min = rand.randint(0,10)
    x_max = 64 - (10 - x_min)
    y_min = rand.randint(0,10)
    y_max = 64 - (10 - y_min)
    return img[y_min:y_max, x_min:x_max]
```
![Randomly Crop Several Smaller Images](images/PipelineResizedRandom.png)  
##### Subtract the Mean of the Image
Recall that the images are stored as numpy arrays.  
Thus, we can compute the mean of the numpy array, then subtract the mean from each index of the array.  
Note that, after having done this, we will have negative values in the numpy array.  
Thus, while the numpy array will still represent an image, it can't be displayed as an image without first handling the negatives in some way.  
```python
def subtractMean(img):
    '''
    Given img, a numpy array representing an image,
    subtracts the mean from the numpy array and returns
    the mean-subtracted result
    '''
    return img - np.mean(img)
```
![64 x 64 Image with Mean Subtracted](images/PipelineSubtractMean.png)  
#### Convolutional Neural Net
We will be building a neural net to output the digits from input images.  The architecture will try to follow, as closely as possible, that architecture described by Goodfellow et al., below:
>Our best architecture consists of eight convolutional hidden layers, one locally connected hidden layer, and two densely connected hidden layers. All connections are feedforward and go from one layer to the next (no skip connections).  
>
The first hidden layer contains maxout units (Goodfellow et al., 2013) (with three filters per unit) while the others contain rectifier units (Jarrett et al., 2009; Glorot et al., 2011).  
> 
The number of units at each spatial location in each layer is [48, 64, 128, 160] for the first four layers and 192 for all other locally connected layers. The fully connected layers contain 3,072 units each.  
>
Each convolutional layer includes max pooling and subtractive normalization. The max pooling window size is 2 × 2. The stride alternates between 2 and 1 at each layer, so that half of the layers don’t reduce the spatial size of the representation.  
>
All convolutions use zero padding on the input to preserve representation size. The subtractive normalization operates on 3x3 windows and preserves representation size.  
>
All convolution kernels were of size 5 × 5. We trained with dropout applied to all hidden layers but not the input.

Below is a sketch of the model architecture that best comports with this description:
```
(0) input (54 x 54 x 3 image)
(1) same-pad 5 × 5 conv  [48] -> 2 × 2 max pooling (stride 2) -> 3 × 3 subtractive normalization -> dropout -> 3-filter maxout
(2) same-pad 5 × 5 conv  [64] -> 2 × 2 max pooling (stride 1) -> 3 × 3 subtractive normalization -> dropout -> ReLU
(3) same-pad 5 × 5 conv [128] -> 2 × 2 max pooling (stride 2) -> 3 × 3 subtractive normalization -> dropout -> ReLU
(4) same-pad 5 × 5 conv [160] -> 2 × 2 max pooling (stride 1) -> 3 × 3 subtractive normalization -> dropout -> ReLU
(5) same-pad 5 × 5 conv [192] -> 2 × 2 max pooling (stride 2) -> 3 × 3 subtractive normalization -> dropout -> ReLU
(6) same-pad 5 × 5 conv [192] -> 2 × 2 max pooling (stride 1) -> 3 × 3 subtractive normalization -> dropout -> ReLU
(7) same-pad 5 × 5 conv [192] -> 2 × 2 max pooling (stride 2) -> 3 × 3 subtractive normalization -> dropout -> ReLU
(8) same-pad 5 × 5 conv [192] -> 2 × 2 max pooling (stride 1) -> 3 × 3 subtractive normalization -> dropout -> ReLU
(9) flatten
(10) fully-connected [3072] -> dropout
(11) fully-connected [3072] -> dropout
(12) output
```
Note that the output itself is relatively complex (see image below), and will be dealt with in more detail after the hidden layers.
![Goodfellow et al. Neural Net Architecture](images/architecture.png)  
Note that the 128 x 128 x 3 referred to above is for the image processing pipeline for the *private* SVHN dataset, to which only Google has access.  
For the *public* SVHN dataset, that portion of the above graphic should read 54 x 54 x 3, in accordance with the preprocessing pipeline which Goodfellow et al. define in their section 5.1, which this notebook attempts to recreate.  

This model architecture will be implemented in `TensorFlow`, using `Keras` as a front-end to aid in layer construction and management.

##### The "Goodfellow et al." Model
```python
img_channels = 3
img_rows = 54
img_cols = 54

# Layer 0: Input
x = Input((img_rows, img_cols, img_channels))

# Layer 1: 48-unit maxout convolution
y = Convolution2D(nb_filter = 48, nb_row = 5, nb_col = 5, border_mode="same", name="1conv")(x)
y = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), border_mode="same", name="1maxpool")(y)
# y = SubtractiveNormalization((3,3))(y)
y = Dropout(0.25, name="1drop")(y)
# y = MaxoutDense(output_dim = 48, nb_feature=3)(y)
y = Activation('relu', name="1activ")(y)

# Layer 2: 64-unit relu convolution
y = Convolution2D(nb_filter = 64, nb_row = 5, nb_col = 5, border_mode="same", name="2conv")(y)
y = MaxPooling2D(pool_size = (2, 2), strides = (1, 1), border_mode="same", name="2maxpool")(y)
# y = SubtractiveNormalization((3,3))(y)
y = Dropout(0.25, name="2drop")(y)
y = Activation('relu', name="2activ")(y)

# Layer 3: 128-unit relu convolution
y = Convolution2D(nb_filter = 128, nb_row = 5, nb_col = 5, border_mode="same", name="3conv")(y)
y = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), border_mode="same", name="3maxpool")(y)
# y = SubtractiveNormalization((3,3))(y)
y = Dropout(0.25, name="3drop")(y)
y = Activation('relu', name="3activ")(y)

# Layer 4: 160-unit relu convolution
y = Convolution2D(nb_filter = 160, nb_row = 5, nb_col = 5, border_mode="same", name="4conv")(y)
y = MaxPooling2D(pool_size = (2, 2), strides = (1, 1), border_mode="same", name="4maxpool")(y)
# y = SubtractiveNormalization((3,3))(y)
y = Dropout(0.25, name="4drop")(y)
y = Activation('relu', name="4activ")(y)

# Layer 5: 192-unit relu convolution
y = Convolution2D(nb_filter = 192, nb_row = 5, nb_col = 5, border_mode="same", name="5conv")(y)
y = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), border_mode="same", name="5maxpool")(y)
# y = SubtractiveNormalization((3,3))(y)
y = Dropout(0.25, name="5drop")(y)
y = Activation('relu', name="5activ")(y)

# Layer 6: 192-unit relu convolution
y = Convolution2D(nb_filter = 192, nb_row = 5, nb_col = 5, border_mode="same", name="6conv")(y)
y = MaxPooling2D(pool_size = (2, 2), strides = (1, 1), border_mode="same", name="6maxpool")(y)
# y = SubtractiveNormalization((3,3))(y)
y = Dropout(0.25, name="6drop")(y)
y = Activation('relu', name="6activ")(y)

# Layer 7: 192-unit relu convolution
y = Convolution2D(nb_filter = 192, nb_row = 5, nb_col = 5, border_mode="same", name="7conv")(y)
y = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), border_mode="same", name="7maxpool")(y)
# y = SubtractiveNormalization((3,3))(y)
y = Dropout(0.25, name="7drop")(y)
y = Activation('relu', name="7activ")(y)

# Layer 8: 192-unit relu convolution
y = Convolution2D(nb_filter = 192, nb_row = 5, nb_col = 5, border_mode="same", name="8conv")(y)
y = MaxPooling2D(pool_size = (2, 2), strides = (1, 1), border_mode="same", name="8maxpool")(y)
# y = SubtractiveNormalization((3,3))(y)
y = Dropout(0.25, name="8drop")(y)
y = Activation('relu', name="8activ")(y)

# Layer 9: Flatten
y = Flatten()(y)

# Layer 10: Fully-Connected Layer
y = Dense(3072, activation=None, name="fc1")(y)

# Layer 11: Fully-Connected Layer
y = Dense(3072, activation=None, name="fc2")(y)

length = Dense(7, activation="softmax", name="length")(y)
digit1 = Dense(11, activation="softmax", name="digit1")(y)
digit2 = Dense(11, activation="softmax", name="digit2")(y)
digit3 = Dense(11, activation="softmax", name="digit3")(y)
digit4 = Dense(11, activation="softmax", name="digit4")(y)
digit5 = Dense(11, activation="softmax", name="digit5")(y)

model = Model(input=x, output=[length, digit1, digit2, digit3, digit4, digit5])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_val = [y0_test, y1_test, y2_test, y3_test, y4_test, y5_test]

model.fit(X_train,
          [y0_train, y1_train, y2_train, y3_train, y4_train, y5_train],
          validation_data=(X_test,
                           y_val),
          nb_epoch=10,
          batch_size=200,
          verbose=1)
```
Note above that some portions of the layers described by Goodfellow et al., e.g. the "subtractive normalization" and the "maxout", have not been implemented in the above model.

That is largely because *how* to implement such features has escaped me, despite [reading more about them](http://www.jmlr.org/proceedings/papers/v28/goodfellow13.pdf) and [engaging the Udacity machine learning community](https://discussions.udacity.com/t/goodfellow-et-al-2013-architecture/202363/2) on this topic.

This is not to say that these features are impossible to implement (indeed, I believe that Goodfellow et al. did actually implement them), but rather to note that, in the absence of an open contribution by Goodfellow et al. to the machine learning community, the effort at recreating them is akin to reinventing them, and that effort is a difficult one.

I have, however, inserted those features into my model where I estimate that they *would* go, if they were implemented.  Of course, this is just a rough estimate, but it is preferable to excluding these features altogether from my implementation.


### Refinement
The [discussion of](https://discussions.udacity.com/t/goodfellow-et-al-2013-architecture/202363) Goodfellow et al.'s architecture is representative of the refinements which I made to the model, prior to its implementation.

Largely, this represents efforts to parse the language of Goodfellow et al. into a viable model architecture.

Another attempted refinement to the model was to share weights between the digit classifiers on the output layer.  My hypothesis was that each digit classifier is actually completing the same task, and so there is inherent waste in training multiple different classifiers to do the same thing (especially since the 4th and 5th digit classifiers would be expected to output 'no digit' the vast majority of the time).

While I still assert that there should be a single digit classifier in the output layer, my implementation of this idea was lacking in the attempted refinement: Each digit classifier output the same class for each image.

This makes a good deal of sense, and one need only consult Goodfellow et al.'s convolutional neural net output layer sketch in the [section](#convolutional-neural-net) above to see why:  Each digit classifier is being fed the same transformed image as input.  That is, for a given H, it is ludicrous to expect a classifier to output one class, and then expect it to output a different class for the exact same H.

Unfortunately, then, I was unable to make this refinement work, and I would suggest that the branching in the neural net would need to occur earlier than the digit classifiers, so that (in theory) the model can learn to pass the unified digit classifier the portion of the image which is relevant for it to classify.  That way, we can train the output layer to recognize digits, while allowing the hidden layers to complete the segmentation and localization tasks.
## Results
### Model Evaluation and Validation
#### The "Goodfellow et al." Model
The evaluation output on the model meant to approximate Goodfellow et al. is as follows:
```
loss        33.787814994744288
length_loss  5.8117896247282284
digit1_loss 14.113818473979387
digit2_loss 11.330030459674594
digit3_loss  2.3841670357126388
digit4_loss  0.14677591633234982
digit5_loss  0.0012344748806895002
length_acc   0.63942454850633534
digit1_acc   0.12434955617001854
digit2_acc   0.29706152432512939
digit3_acc   0.852081420244994
digit4_acc   0.99089378636657621
digit5_acc   0.99992347719620445
```
Notably, whole-sequence transcription accuracy was `0.0%`.

First, let's discuss the length accuracy.  
Note from above that ~18,000 of the 33,403 (or ~54%) of the training examples had a length of `2`.  
Thus, it's within the realm of possibility that 63.9% of the testing examples also had a length of `2`, and that the model is simply always guessing `2`.  
I am not convinced that the model learned how to determine the length representation.

Next, let's discuss digit1 accuracy.  
At ~12.4%, it's completely within the realm of possibility that the model is simply guessing the same number for each image it sees.  This accuracy is almost certainly not better than random chance, which strongly suggests that the model did not learn how to represent the first digit of an image.

Next, let's discuss the remaining digit accuracies.
Recall that very few examples in the training set had a fifth or fourth digit.  
Accordingly, it is almost certainly the case that the fourth and fifth digit classifiers are simply guessing, for every image they see, "no digit."  
This is probably occurring with the third digit as well.  Although ~27% of the training examples had a third digit, it seems possible at least that only 15% of testing examples were longer than two digits, meaning that if this classifier was just guessing "no digit," it would have achieved an accuracy of 85%.
While it is tempting to think that perhaps the model learned *some* way of representing the second digit in the image, albeit poorly, it is just as possible that it is guessing the same digit each time, and there just happens to be a second digit which occurs with ~30% frequency in the testing set.

Overall, the performance of this model is disappointing, especially considering its depth of representation.  
It is unclear which architectural changes could be taken to improve the performance of this model.  

Accordingly, for comparison, I also tested the following model from a Google Groups [discussion](https://groups.google.com/forum/#!topic/keras-users/UIhlW423YFs).

#### The "Ritchie Ng" Model
```python
x = Input((img_rows, img_cols, img_channels))
y = Convolution2D(32, 3, 3, border_mode='same')(x)
y = Activation('relu')(y)
y = Convolution2D(32, 3, 3, border_mode='same')(y)

y = Activation('relu')(y)
y = MaxPooling2D((2,2), strides=(2,2))(y)
y = Dropout(0.5)(y)

y = Flatten()(y)
y = Dense(1024, activation="relu")(y)
length = Dense(7, activation='softmax')(y)
digit_1 = Dense(11, activation='softmax')(y)
digit_2 = Dense(11, activation='softmax')(y)
digit_3 = Dense(11, activation='softmax')(y)
digit_4 = Dense(11, activation='softmax')(y)
digit_5 = Dense(11, activation='softmax')(y)
branches = [length, digit_1, digit_2, digit_3, digit_4, digit_5]
model = Model(input=x, output=branches)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
history = model.fit(X_train,
          [y0_train, y1_train, y2_train, y3_train, y4_train, y5_train],
          validation_data=(X_test,
                           y_val),
          nb_epoch=10,
          batch_size=200,
          verbose=1)
```
Unfortunately, performance on this model was similar to the performance on the Goodfellow et al. model:
```
  Training:
loss        35.911776717912154
length_loss  7.3694877039778772
digit1_loss 11.65790105544124
digit2_loss 12.014987309202734
digit3_loss  4.2729705319234146
digit4_loss  0.59208743797327312
digit5_loss  0.0043430576070498062
length_acc   0.54278186844977128
digit1_acc   0.27671995566645163
digit2_acc   0.25456559317084204
digit3_acc   0.73489611498303531
digit4_acc   0.9632656717006336
digit5_acc   0.99973055531354615

  Testing:
loss        31.235912252939329
length_loss  5.8117910404981803
digit1_loss 11.561913261436928
digit2_loss 11.330033568238351
digit3_loss  2.3841658736866185
digit4_loss  0.1467749148807943
digit5_loss  0.0012335209788844094
length_acc   0.6394245511244151
digit1_acc   0.28267523494013452
digit2_acc   0.29706152324870633
digit3_acc   0.85208142352899641
digit4_acc   0.99089379397451527
digit5_acc   0.99992347726918229

Whole-Sequence Accuracy: 0.0%
```
Notable, however, is the difference between the digit1 accuracies in the Goodfellow et al. model and the Ritchie Ng model.

It must be the case that, however poorly, the Ritchie Ng model has learned how to discern the first digit of the numbers with better than random chance.  While that's not a sweeping victory, it is at least an interesting result of this model.
### Justification
The benchmark established above for this problem is to achieve whole-sequence transcription accuracy of at least 96%.

With both the Goodfellow et al. and the Ritchie Ng models, the actual whole-sequence accuracy achieved was 0.0%.

Accordingly, it is not the case that either model can be said to have solved the problem adequately.

## Conclusion
### Free-Form Visualization
### Reflection
#### Goodfellow et al.
Unfortunately, I was unable to reproduce the performance obtained by [Goodfellow et al.](https://arxiv.org/pdf/1312.6082v4.pdf), namely a whole-sequence transcription accuracy of ~96% on the public SVHN dataset.

I am, however, able to identify factors which could contribute to the discrepancy between their performance and mine:  
* Architectural Differences  
   * The output layer that Goodfellow et al. use, namely six separate softmax classifiers, differs from my mine.  Specifically, the *n*th classifier does **not** backpropagate when the *n*th digit is not present, however classifiers backpropagate on every input.  There could be some clever way of "turning off" the classifiers when they're not supposed to be active, but I haven't found it.  
   * Maxout layers are not present in my architecture, although their presence is explicitly noted by Goodfellow et al.  This is largely because I [could not discern](https://discussions.udacity.com/t/goodfellow-et-al-2013-architecture/202363/5) what was meant by "maxout layer".  That's not to assert that it's impossible to understand, but rather to convey that I didn't understand.  
   * Subtractive normalization is not present in my architecture, although (perhaps unrelated except by semantics) I do subtract the mean of each image during preprocessing before it is passed to my architecture as input, as did Goodfellow et al.  This absence is also attributable to [not knowing](https://discussions.udacity.com/t/goodfellow-et-al-2013-architecture/202363/5) what was meant by "subtractive normalization," in context.  As with maxout layers, I mean in this bullet point to convey more that I didn't understand, more than to assert that no one could understand.  
* Library Differences
   * For this project, I used the `Keras` frontend to Google's `TensorFlow` library, as that library is quickly becoming established in the Deep Learning community.  
   * For their paper, Goodfellow et al. used the `DistBelief` implementation of a neural net.  Although [recreations](http://alexminnaar.com/implementing-the-distbelief-deep-neural-network-training-framework-with-akka.html) of `DistBelief` exist, and although it's possible to view `TensorFlow` as [an extension](https://en.wikipedia.org/wiki/TensorFlow#DistBelief) of `DistBelief`, `DistBelief` itself is not publicly available.  
* Training Details
   * For this project, I arbitrarily trained for 10 epochs within `Keras`.  I am fairly certain that this is not comparable to the "six days" of training conducted by Goodfellow et al. on "10 replicas" within `DistBelief`, but I also don't know how I could make my training regimen look closer to theirs.
   * Although Goodfellow et al. describe their image preparation steps in laudable detail, the exact representation of their images and labels (e.g. `numpy` arrays, with one-hot-encoded labels) is left out of their paper.  As such, I've had to guess at a representation that would work.  One key difference is that "no digit" is an explicit class in my representation, which the individual classifiers need to correctly guess.  This is a departure from the Goodfellow et al. training, in which unneeded classifiers were turned on or off based on the output of the length classifier.
   * The loss function used by Goodfellow et al. is not stated, although they do hint that "one can maximize [logarithmic digit probability] on the training set using a generic method like stochastic gradient descent".  Thus, my implementation has arbitrarily chosen to use the AdamOptimizer on sparse_categorical_crossentropy.
   
Given the number of identified differences above, it is almost not surprising that my recreation of the Goodfellow et al. model was unable to reproduce their performance.  

#### Ritchie Ng
However, fewer such differences exist between my recreation of the [Ritchie Ng](https://groups.google.com/forum/#!topic/keras-users/UIhlW423YFs) neural net and my own, and my [implementation](#Trying-a-Random-Model-from-the-Internet) of that model was also unable to reproduce its performance.  
While that model has a whole-sequence transcription accuracy of 0%, it notably [achieved](https://groups.google.com/d/msg/keras-users/UIhlW423YFs/e2RL2p5hFAAJ) a ~92% first-digit accuracy, and an ~84% second-digit accuracy.

However, while the exact `Keras` calls used to build the model described by Ritchie Ng was provided, his preprocessing scheme, and the representation he chose for his data, including the labels, was *not provided*.  This discrepancy, I would assert, leaves enough room between his experiment and mine so as to greatly hinder comparing the two.

### Improvement
I suspect that, somewhere in this lengthy pipeline of SVHN images to neural net classifications, is a misrepresentation, a mistaken assumption, an incomplete implementation, or a combination of these, that, if identified and fixed, would allow this model to learn much better.

I would assert that the fundamental approach, that of using a deep neural net to output whole sequences of digits, has been [valid for decades](https://www.youtube.com/watch?v=FwFduRA_L6Q).  
The trouble at present is in designing a model architecture that can output whole sequences of digits as they appear in the wild, with all the visual artifacts, distortions, and extra information those wild images contain.  
As the motivation for undertaking this project, I would assert that Goodfellow et al. have solved this problem computationally, but that their implementation, being private, cannot easily be recreated.

However, to somewhat mirror the [above section](#Goodfellow-et-al.), I think the following improvements would greatly benefit this model:  
* Implementing an output layer that turns its classifiers on and off depending on how many digits it detects in the image.  This was, in my implementation, much more easily said than done, but if it could be achieved, it would likely help the model not to learn useless features, or resort to guessing.
* Identifying an ideal representation of the image labels.  This information is readily available (if somewhat cumbersome to work with), but it's unclear to me which representation (e.g. one-hot, continuous, etc.) would be ideal to feed to the neural net.
* Identifying an ideal representation of the input images.  That is, what should a given image look like after pre-processing?  Not just "what type of preprocessing should be performed," but also "what structure (e.g. `numpy` array) should this image be stored in," and "what is the shape of that structure?"
