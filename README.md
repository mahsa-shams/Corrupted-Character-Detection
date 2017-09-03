# Corrupted Character Detection of car License Plates

## Objectives
* Detection of healthy characters
* Detection of corrupted characters

## Description
There can be two type of inputs, cropped license plate (grayscale) and images of back of cars (colored). In second group, our first process is to detect license plates. This can be done using object detection techniques (with higher accuracy) or opencv shape detection methods (lower accuracy). since I didn't have enough time, I worked on just first group images. 

The common pipline of object detection includes these steps:
1. Preprocess
2. Feature Extraction
3. Classification

There are many methods for each of these steps, from traditional methods to deep learning based ones. since I hadn't enough annotated training image and time(!), I applied traditional methods which I think they can perform the job fine. 

The preprocess step includes image resizing, conversion to grayscale and thresholding. I used HOG (Histogram of Oriented Gradient) to compute feature vectors, which represent the occurrence
of gradient orientations in the image. Plate characters are composed of strong geometric shapes and high-contrast edges that encompass a range of orientations, so HOG features could be very suitable to represent them. 
I used contours of the input image to extract probable regions of characters. then, I proned them with their width (as character can be in a width range). If the extracted region width is bigger than character max width threshold, there should be a corrupted character or more than one character. sometimes plate characters stick together due to getting dirty or extra marks on plate. so I threshold image with a degree close to characters black to increase the chance of character contour separation. I didn't do this step (thresholding) for every region because corrupted characters are in many different gray levels so we may loose them in this step. 
After finding regions with contour extraction and prunning them with the width range, HOG descriptor computation is perfomed and next step is to classify them. 
I used SVM classification method to classify obtained regions. SVM method is fast, highly
accurate, and less prone to overfitting compared to many other classification methods.
This solution runs very fast (about 1 ms for each plate) and perform the job with acceptable accuracy. There are some extra rectangles on the output which I didn't have time to fix them!


### Code pipline:

1. load train dataset and their labels
* positive data which is number and characters of our plates
* negative data which is damaged characters

2. preprocess train set: since train set are croped images, preprocess stage for them includes color adjusments and resize

3. preprocess test set: testset is plates images, so their preprocess stage includes color adjusments, resize, character segmentation and classification based on width

4. feature extraction

5. character classification

6. write output


### Running Code
Qt and Opencv3 is needed for running the code. There is a trained model (model4.yml) which is trained using 3706 persian positive characters and 1710 negative (corrupted character) images. 
You can run the code with one string path of testset to test them and with two pathes to train and test(first train path containing two directories of positive and negative images).  
