#**Traffic Sign Recognition** 

[//]: # (Image References)

[distribution]: ./writeup_report_images/Distribution_of_traffic_signs "Distribution of traffic signs in the training set"
[top5]: ./writeup_report_images/top_5 "Top 5 classes in training set"
[flop5]: ./writeup_report_images/flop_5 "The 5 classes in training set with the lowest occurence"
[sign1]: ./new_images/id_1.jpeg "Traffic Sign 1"
[sign2]: ./new_images/id_2.jpg "Traffic Sign 2"
[sign3]: ./new_images/id_13.jpg "Traffic Sign 3"
[sign4]: ./new_images/id_17.jpeg "Traffic Sign 4"
[sign5]: ./new_images/id_27.png "Traffic Sign 5"
[graySign1]: ./writeup_report_images/grayscaled_1.png "Grayscaled Traffic Sign 1"
[graySign2]: ./writeup_report_images/grayscaled_2.png "Grayscaled Traffic Sign 2"
[graySign3]: ./writeup_report_images/grayscaled_13.png "Grayscaled Traffic Sign 3"
[graySign4]: ./writeup_report_images/grayscaled_17.png "Grayscaled Traffic Sign 4"
[graySign5]: ./writeup_report_images/grayscaled_27.png "Grayscaled Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

##Writeup report from Michael Bachmann

You're reading my writeup report. And here is a link to my [project code](https://github.com/mbachm/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

The writeup is structured on the basis of the rubrics. First, it starts with the dataet exploration follwed by the design, architecture and test section of the neural network. The last part is about the test with the new images.


###1. Data Set Summary & Exploration

####1.1. Dataset Summary

The provided dataset has the following structure:

| Dataset                                  | Description | 
|:----------------------------------------:|:-----------:| 
| Size of training set                     | 34799       |
| Size of validation set                   |  |
| Size of testing set                      | 12630       |
| Number of unique classes/labels          | 43          |
| Original shape of the images in the sets | (32, 32, 3) |

You can find the code for this section in the second code cell. The first code cell just load the provided data sets.


####1.2. Exploratory Visualization

First, I provide a bar chart which shows the distribution of the unique classes in the training set. You can find the code for this distribution in the fourth code cell in the jupyter notebook.

![Distribution of the unique classes in the training set][distribution]

You can see that the images are not distributed equally. The occurance of some traffic signs is more than 10 times higher than the occurence of the sign *speed limit (20 km/h)*. The sign with the highest occurence is *speed limit (50 km/h)*.

![Top 5 traffic signs][top5]
This images shows the top 5 signs that are present in the data set. The signs are:
- Id 2: Speed limit (50 km/h)
- Id 1: Speed limit (30 km/h)
- Id 13: Yield
- Id 12: Priority road
- Id 38: Keep right

![The 5 classes in training set with the lowest occurence][flop5]
This images shows the 5 signs have the lowest occurence in the test set. 
- Id 0: Speed limit (20 km/h)
- Id 37: Go straight or left
- Id 19: Dangerous curve to the left
- Id 32: End of all speed and passing limits
- Id 27: Pedestrians

You can find the code for the bars in the fifth and sixth code cell. Furthermore, you can see images for each of the above signs below the cell. All images are from the provided data sets.


###2. Design and Test a Model Architecture

####2.1. Preprocessing

You can find the preprocessing under the headline *Pre-process the Data Set (normalization, grayscale, etc.)*, the seventh code cell in the jupyter notebook.
The preprocessing for the sets is done in the function `normalize_and_greyscale_image_set`. First, it converts the RGB image into a greyscale image. This is done beacuse I wanted my network to focus on the shape of the traffic sign and not the color it has. Second, the preprocessing normalizes the images with the function `normalize_greyscaled_Image`. This way I want to achieve that my NN focus on the underlying distributions/traffic sign shape.
Afterwards, I add an additional dimension to the image. This is done to get it in the shape (32, 32, 1), as after the grayscaling of the image with `cv2.cvtColor`the dimension is removed.

Right below the functions you can see that my code is converting the images shape from (32, 32, 3) to (32, 32, 1) for all given sets.
```
Image Shape: (32, 32, 3)
Normalized image shape: (32, 32, 1)
Normalized validation set image shape: (32, 32, 1)
Normalized validation set image shape: (32, 32, 1)
```

####2.2. Model Architecture

My architecture is inspired by the LeNet architecture from the provided LeNet jupyter notebook. You can find my architecture in the tenth code cell of the jupyter notebbok. The name of the function is `traffic_sign_classifier`. It consists of 6 layer, which all have a activation function except the last layer. The first 3 layers are convolutional, the last 3 are fully connected. There are 2 pooling layers for the convolutional part, a flatten layer and 2 dropout layers.
The arguments used for tf.truncated_normal are `mu = 0` and `sigma = 0.1`. The dropout rate for both dropout layers is `0.65`. I tried out a lower dropout rate (0.6), but this did not make my network better. With a dropout rate of 0.5 my network was not able to learn. With a higher dropout rate, I was fearing to overfit my network to the training data.

Before I got my final mode, I tried a simpler one with one convolution less. But this network did not perform as well as this one. I also tried out different dimension for the layers, e.g. 10 dimensions for the first layer and a dimension of 56 for the last convolutional. But this network did also not work as expected. With my final archticeture I found a solution to achieve my targets without making the network to complex.

Below you can see the layers of my final model:

| Layer         				|     Description	        					| 
|:-----------------------------:|:---------------------------------------------:| 
| Input         				| 32x32x3 RGB image   							| 
| Layer 1: Convolution 5x5 		| 1x1 stride, VALID padding, outputs 28x28x8	|
| RELU							|												|
| VALID pooling	      			| 2x2 stride, outputs 14x14x8					|
| Layer 2: Convolution 4x4 		| 1x1 stride, VALID padding, outputs 12x12x26   |
| Dropout						| Rate: 0.65   									|
| VALID pooling					| 2x2 stride, outputs 6x6x26					|
| Layer 3: Convolution 2x2 		| 1x1 stride, VALID padding, outputs 4x4x48		|
| RELU							|												|
| FLATTEN						| Output = 768									|
| Layer 4: Fully connected 		| Output = 400        							|
| Dropout						| Rate: 0.65   									|
| Layer 5: Fully connected 		| Output = 120        							|
| RELU							|												|
| Layer 6: Fully connected 		| Output = 43       							|


####2.3. Model Training

The choosen hyper parameters are `EPOCHS = 400` and `BATCH_SIZE = 128`. With 400 epochs I was able to gain a good validation and test result for my trained network. I tried out 500 epochs, but my network did not perform better. With less epochs (200, 250, 300 and 350), I was not able to achieve a good validation result. I already have good experiences with a batch size of 128, so I didn't change it. They can also be found in the tenth code cell.

To train the model, I used an AdamOptimizer with a learning rate of `rate = 0.00008` (see eleventh code cell). I tried out many learning rates from 0.001 to 0.00001, but the rate 0.00008 did perform best in my tests. In the same cell, you also see that I used `softmax_cross_entropy_with_logits` and `reduce_mean` functions from Tensorflow to define my loss function, which the AdamOptimizer uses.

The folling code cell defines the `evaluate`function. I used the one provided from the LeNet session, so I won't explain this part.

The thirteenth code cell contains my training function. First, I shuffle my normalized training set and the corresponding y-label. Afterwards, the model is trained with the provided batch size. Afterwards, the accuracy is calculated with the `evaluate` function from above. This is done for 400 epochs, which you can see in the printout of the code cell. The validation accuracy of the last epoch is 0.948.

####2.4. Solution Approach
In the last code cell of the section *Step 2: Design and Test a Model Architecture* you can see the calculated accuracy of the given test set. My network was able to gain a accuracy of 93,4%.


###3. Test a Model on New Images

####3.1. Acquiring New Images

I acquired 5 new images from the internet. You can see the resized and grayscaled images in the table below. The second code cell in the section *Step 3: Test a Model on New Images* of the notebook transforms the signs into grayscaled ones.

| Image ID 	| Sign name 			| Sign image 				| grayscaled sign 				|
|:---------:|:---------------------:|:-------------------------:|:-----------------------------:|
| 1  		| Speed limit (30 km/h) | ![Traffic sign 1][sign1]	| ![Traffic sign 1][graySign1]	|
| 2  		| Speed limit (50 km/h) | ![Traffic sign 2][sign2]	| ![Traffic sign 2][graySign2]	|
| 13 		| Yield					| ![Traffic sign 3][sign3]	| ![Traffic sign 3][graySign3]	|
| 17 		| No entry 				| ![Traffic sign 4][sign4]	| ![Traffic sign 4][graySign4]	|
| 27 		| Pedestrians			| ![Traffic sign 5][sign5]	| ![Traffic sign 5][graySign5]	|

####3.2. Performance on New Images



####3.3. Model Certainty - Softmax Probabilities



####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 