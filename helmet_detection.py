# **Problem Statement**

# **Business Context**

# Workplace safety in hazardous environments like construction sites and industrial plants is crucial to prevent accidents
# To overcome these challenges, SafeGuard Corp plans to develop an automated image analysis system capable of detecting wh

# **Objective**

# As a data scientist at SafeGuard Corp, you are tasked with developing an image classification model that classifies imag
# - **With Helmet:** Workers wearing safety helmets.
# - **Without Helmet:** Workers not wearing safety helmets.

# **Data Description**

# The dataset consists of **631 images**, equally divided into two categories:
# - **With Helmet:** 311 images showing workers wearing helmets.
# - **Without Helmet:** 320 images showing workers not wearing helmets.
# **Dataset Characteristics:**
# - **Variations in Conditions:** Images include diverse environments such as construction sites, factories, and industria
# - **Worker Activities:** Workers are depicted in different actions such as standing, using tools, or moving, ensuring ro

# **Installing and Importing the Necessary Libraries**

!pip install tensorflow[and-cuda] numpy==1.25.2 -q  #install tensorflow

import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.__version__)

# **Note:**
# - After running the above cell, kindly restart the notebook kernel (for Jupyter Notebook) or runtime (for Google Colab) 
# - On executing the above line of code, you might see a warning regarding package dependencies. This error message can be

import os                   #import os to interact with Operating system.
import random                #importing random module for generating psuedo-random numbers and making random choices
import numpy as np                  # Importing numpy for Matrix Operations
import pandas as pd                  # Importing pandas to read CSV files
import seaborn as sns             # Importing seaborn to plot graphs

import matplotlib.image as mpimg   # Importting matplotlib for Plotting and visualizing images
import matplotlib.pyplot as plt
import math              # Importing math module to perform mathematical operations
import cv2             # Importing openCV for image processing


# Tensorflow modules
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator        # Importing the ImageDataGenerator for data augmentation
from tensorflow.keras.models import Sequential                                                   # Importing the sequential module to define a sequential model
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization # Defining all the layers to build our CNN Model
from tensorflow.keras.optimizers import Adam,SGD                                                 # Importing the optimizers which can be used in our model
from sklearn import preprocessing                                                                # Importing the preprocessing module to preprocess the data
from sklearn.model_selection import train_test_split                                             # Importing train_test_split function to split the data into train and test
from sklearn.metrics import confusion_matrix  # Importing confusion_matrix to plot the confusion matrix
from tensorflow.keras.models import Model
from keras.applications.vgg16 import VGG16

# Display images using OpenCV
# from google.colab.patches import cv2_imshow  # Remove if not using Colab

#Imports functions for evaluating the performance of machine learning models
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score, recall_score, precision_score, classification_report
from sklearn.metrics import mean_squared_error as mse                                                 # Importing cv2_imshow from google.patches to display images

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# **Observation:**
# All the python libraries are imported successfully.

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
tf.keras.utils.set_random_seed(812)

# **Observation:**
# Seed is set for Numpy, backend and python.

# **Data Overview**

# Loading the data

# uncomment and run the below code snippets if the dataset is present in the Google Drive
# from google.colab import drive  # Remove if not using Colab
# drive.mount('/content/drive')  # Remove if not using Colab

# Load the image file of the dataset
images = np.load('/content/drive/MyDrive/Colab Notebooks/images_proj.npy')

# Load the labels file of the dataset
labels = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Labels_proj.csv')

# **Observation:**
# 1. Image variable hold the array of the image data
# 2. Labels variable hold the Dataframe containing the corresponding lables.

# **Exploratory Data Analysis**

# Plot random images from each of the classes and print their corresponding labels.

print(images.shape)   #print dimension of the loaded image
print(labels.shape)   #print shape of the label variable

# **Observation:**
# There are 631 RGB images of shape 200 x 200 X 3, each image having 3 channels.

plt.imshow(images[4]);  #display image

# **Observation:**
# Image which is located at 4th index is displayed

plt.imshow(images[10]);  #display image

# **Observation:**
# Image which is located at 10th index is displayed

def plot_images(images,labels):
    keys=dict(labels['Label'])
    rows = 3     # Defining number of rows=3
    cols = 4     # Defining number of columns=4
    fig = plt.figure(figsize=(10, 8))   # Defining the figure size to 10x8
    for i in range(cols):
        for j in range(rows):
            random_index = np.random.randint(0, len(labels))   # Generating random indices from the data and plotting the images
            ax = fig.add_subplot(rows, cols, i * rows + j + 1)  # Adding subplots with 3 rows and 4 columns
            ax.imshow(images[random_index, :])    # Plotting the image
            ax.set_title(keys[random_index])  #set title
    plt.tight_layout() # Add this line
    plt.show()

plot_images(images,labels)  #display image with their labels

# **Observation:**
# Here the image with Lable 1 shows that they are with Helmet. And where the lable is 0 shows that they are without helmet

# Checking for class imbalance

# Create a count plot
plt.figure(figsize=(6, 4))
ax = sns.countplot(x=labels.iloc[:, 0], palette=['red', 'green'])

# Add exact counts on top of bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10, )

# Add labels
plt.xlabel("Class Labels", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)
plt.title("Count of Images per Class", fontsize=14)
plt.xticks(ticks=[0, 1], labels=["Without Helmet (0)", "With Helmet (1)"])

# Show plot
plt.show()

# **Observation:**
# It shows that there are 320 such image which doesnot include any helmet . And there are 311 such image which has helmet.

# **Data Preprocessing**

# Converting images to grayscale

# Function to plot the original and processed images side by side
def grid_plot(img1,img2,gray=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img1)    #display first image in the first subplot
    axes[0].set_title('Original Image')  #set title of th first subplot to Original image
    axes[0].axis('off')  #turns off the axes labels and making the image the primary focus

    if gray:
      axes[1].imshow(img2,cmap='gray')  #if true, display second image of the second subplot using grayscale cmap
    else:
      axes[1].imshow(img2)  #if false, display second image in the second subplot using default cmap
    axes[1].set_title('Gray Image')  #set title of the second subplot to Gray image
    axes[1].axis('off')  #turns of the axis lables and ticks for the second subplots.

    plt.show()  #display image

#Function to convert an image into Gray image
gray_images = []   #store Grayscale version of the image
for i in range(len(images)):    #start loops and iterate through each images
  gray_img = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)   #perform actual grayscale conversion
  gray_images.append(gray_img)  #add newly created grayscale image

# choosing an image
n = 5
# plotting the original and preprocessed image
grid_plot(images[n],gray_images[n],gray=True)

# **Observation:**
# Image which is located at 5th index is displayed side by side one with Original color image  and the grayscale image

# Splitting the dataset

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(np.array(images),labels , test_size=0.2, random_state=42,stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp , test_size=0.5, random_state=42,stratify=y_temp)

print(X_train.shape,y_train.shape) #print the shape of the train data
print(X_val.shape,y_val.shape) #print the shape of the validation data
print(X_test.shape,y_test.shape) #print the shape of the test data

# Data Normalization

#Since the image pixel values range from 0-255. we divide all the pixel values by 255 to standardize the images to have values between 0-1.
X_train_normalized = X_train.astype('float32')/255.0 #Normalize the training images
X_val_normalized = X_val.astype('float32')/255.0  #Normalize the validation images
X_test_normalized = X_test.astype('float32')/255.0   #Normalize the test images

# **Model Building**

# Model Evaluation Criterion
# Based on the problem statement, the aims is to enhance efficiency, scalability, and accuracy, which ultimately fostering
# Best Choice: Accuracy
# As the dataset is relatively balanced between "With Helmet" and "Without Helmet" categories, accuracy will give a better
# Along with Accuracy we will use precision so that it will show How many of the flagged violators are actually violators?

# Utility Functions

# defining a function to compute different metrics to check performance of a classification model built using statsmodels
def model_performance_classification(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # checking which probabilities are greater than threshold
    pred = model.predict(predictors).reshape(-1)>0.5

    target = target.to_numpy().reshape(-1)


    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred, average='weighted')  # to compute Recall
    precision = precision_score(target, pred, average='weighted')  # to compute Precision
    f1 = f1_score(target, pred, average='weighted')  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame({"Accuracy": acc, "Recall": recall, "Precision": precision, "F1 Score": f1,},index=[0],)

    return df_perf

def plot_confusion_matrix(model,predictors,target,ml=False):
    """
    Function to plot the confusion matrix

    model: classifier
    predictors: independent variables
    target: dependent variable
    ml: To specify if the model used is an sklearn ML model or not (True means ML model)
    """

    # checking which probabilities are greater than threshold
    pred = model.predict(predictors).reshape(-1)>0.5

    target = target.to_numpy().reshape(-1)

    # Plotting the Confusion Matrix using confusion matrix() function which is also predefined tensorflow module
    confusion_matrix = tf.math.confusion_matrix(target,pred)
    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        linewidths=.4,
        fmt="d",
        square=True,
        ax=ax
    )
    plt.show()

# Model 1: Simple Convolutional Neural Network (CNN)

# Initializing Model
model_1 = Sequential()

# Convolutional layers
model_1.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(200,200,3))) #defining the shape of the input image
model_1.add(MaxPooling2D((4, 4), padding='same'))
model_1.add(Conv2D(64, (3, 3), activation='relu', padding="same")) #the number of output channels,the kernel shape and the activation function
model_1.add(MaxPooling2D((2,2), padding='same')) #define the shape of the pooling kernel
model_1.add(Conv2D(128, (3,3), activation='relu', padding="same")) #define the number of output channels,the kernel shape and the activation function

# Flatten and Dense layers
model_1.add(Flatten())
model_1.add(Dense(4, activation='relu'))
model_1.add(Dense(1, activation='sigmoid'))  #define the number of neurons in the output layer and the activation function

# Compile with Adam Optimizer
opt = Adam(learning_rate=0.001) #define the learning rate.
model_1.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy","Precision"])

# Summary
model_1.summary()

# **Observation:**
# The summary shows the layers in the model, Output shape of each layer, and the number of parameters in each layers. The 

history_1 = model_1.fit(
            X_train_normalized, y_train,
            epochs=20, #the number of epochs
            validation_data=(X_val_normalized,y_val),
            shuffle=True,
            batch_size=32, #batch size
            verbose=2
)

# Observation:
# history_1 variables stores information about the training and validation data at each epochs.

plt.plot(history_1.history['accuracy']) #display training accuracy
plt.plot(history_1.history['val_accuracy']) #display validation accuracy
plt.title('Model Accuracy') #Set title
plt.ylabel('Accuracy') #Set label for the y-axis
plt.xlabel('Epoch') #Set label for the x-axis
plt.legend(['Train', 'Validation'], loc='upper left')  #Add legend to plot
plt.show() #display the plot

model_1_train_perf = model_performance_classification(model_1, X_train_normalized,y_train)  #calculates performance classification metrics for training data

print("Train performance metrics")
print(model_1_train_perf)

plot_confusion_matrix(model_1,X_train_normalized,y_train)  #display confusion matrix for training data

model_1_valid_perf = model_performance_classification(model_1, X_val_normalized,y_val) #calculates performance classification metrics for validation data

print("Validation performance metrics")
print(model_1_valid_perf)

plot_confusion_matrix(model_1,X_val_normalized,y_val)  #display confusion matrix for validation data

# **Observation:**
# The model has achieved 100% accuracy on the training set and 100% accuracy on the validation set.Its seems like model is

# Vizualizing the predictions

# For index 4
plt.figure(figsize=(2,2))  #create a new figure
plt.imshow(X_val[4]) #display the image located at index 4
plt.show()   #dispaly the image
prediction = model_1.predict(X_val_normalized[4].reshape(1,200,200,3))  #use trained model to make prediction
predicted_label = prediction[0][0]>0.5  # Extract the predicted class label
print('Predicted Label:', 1 if predicted_label else 0)
# Fix indexing issue in y_val
true_label = y_val.iloc[4]  #retrive actual true label for the image
print('True Label:', true_label)

# For index 10
plt.figure(figsize=(2,2))  #create a new figure
plt.imshow(X_val[10])  #display the image located at index 10
plt.show()  #dispaly the image
prediction = model_1.predict(X_val_normalized[10].reshape(1,200,200,3))  #use trained model to make prediction
predicted_label = prediction[0][0]>0.5  # Extract the predicted class label
print('Predicted Label:', 1 if predicted_label else 0)
# Fix indexing issue in y_val
true_label = y_val.iloc[10]  #retrive actual true label for the image
print('True Label:', true_label)

# **Observation:**
# For both image Predicted label and true label is matching.

# Model 2: (VGG-16 (Base))

# We will be loading a pre-built architecture - VGG16, which was trained on the ImageNet dataset.
# For training VGG16, we will directly use the convolutional and pooling layers and freeze their weights i.e. no training 

vgg_model = VGG16(weights='imagenet',include_top=False,input_shape=(200,200,3)) #Creates an instance of the VGG16 model from the keras module.
vgg_model.summary()   #print summary of the vgg model architecture.

# **Observation:**
# Summary of the vgg_model includes list of layers, output shape and number of parameters in each layers. The total number

# Making all the layers of the VGG model non-trainable. i.e. freezing them
for layer in vgg_model.layers:
    layer.trainable = False

model_2 = Sequential()

# Adding the convolutional part of the VGG16 model
model_2.add(vgg_model)

# Flattening the output of the VGG16 model
model_2.add(Flatten())

# Adding a dense output layer
model_2.add(Dense(1, activation='sigmoid'))

opt=Adam(learning_rate=0.001) #defining the learning rate
# Compile model
model_2.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

# Generating the summary of the model
model_2.summary()

# **Observtaion:**
# Summary of the vgg_model includes list of layers, output shape and number of parameters in each layers. The total number

train_datagen = ImageDataGenerator()  #creates an instance of ImageDataGenerator.

# Epochs
epochs = 20
# Batch size
batch_size = 128

history_2 = model_2.fit(train_datagen.flow(X_train_normalized,y_train,
                                      batch_size=batch_size,
                                      seed=42,
                                      shuffle=False),
                    epochs=epochs,
                    steps_per_epoch=X_train_normalized.shape[0] // batch_size,
                    validation_data=(X_val_normalized,y_val),
                    verbose=1)

# **Observation:**
# history_2 stores both the training and validation data at each epochs.

plt.plot(history_2.history['accuracy']) #Display training accuracy
plt.plot(history_2.history['val_accuracy']) #Display validation accuracy
plt.title('Model Accuracy') #Set title for the plot
plt.ylabel('Accuracy') #Set label for the y-axis
plt.xlabel('Epoch') #Set label for the x-axis
plt.legend(['Train', 'Validation'], loc='upper left') #Add legend
plt.show() #display the plot

model_2_train_perf = model_performance_classification(model_2,X_train_normalized,y_train) #calculates performance  metrics for training data

print("Train performance metrics")
print(model_2_train_perf)

plot_confusion_matrix(model_2,X_train_normalized,y_train) #display confusion matrix for training data

model_2_valid_perf = model_performance_classification(model_2, X_val_normalized,y_val)  #calculates performance metrics for validataion data

print("Validation performance metrics")
print(model_2_valid_perf)

plot_confusion_matrix(model_2,X_val_normalized,y_val) #display validation confusion matrix

# **Observation:**
# The model has achieved 100% accuracy on the training set and 100% accuracy on the validation set.Its seems like model is

# Visualizing the prediction:

plt.figure(figsize=(2,2))  #create a new figure
plt.imshow(X_val[4]) #display the image located at index 4
plt.show()  #dispaly the image
prediction = model_2.predict(X_val_normalized[4].reshape(1,200,200,3)) #use trained model to make prediction
predicted_label = prediction[0][0]>0.5  # Extract the predicted class label
print('Predicted Label:', 1 if predicted_label else 0)
# Fix indexing issue in y_val
true_label = y_val.iloc[4] #retrive actual true label for the image
print('True Label:', true_label)

plt.figure(figsize=(2,2))  #create a new figure
plt.imshow(X_val[10]) #display the image located at index 10
plt.show() #dispaly the image
prediction = model_2.predict(X_val_normalized[10].reshape(1,200,200,3))  #use trained model to make prediction
predicted_label = prediction[0][0]>0.5  # Extract the predicted class label
print('Predicted Label:', 1 if predicted_label else 0)
# Fix indexing issue in y_val
true_label = y_val.iloc[10] #retrive actual true label for the image
print('True Label:', true_label)

# **Observation:**
# For both image Predicted label and true label is matching.

# Model 3: (VGG-16 (Base + FFNN))

model_3 = Sequential()

# Adding the convolutional part of the VGG16 model
model_3.add(vgg_model)

# Flattening the output of the VGG16 model
model_3.add(Flatten())

#Adding the Feed Forward neural network
model_3.add(Dense(256,activation='relu'))
model_3.add(Dropout(rate=0.4))
model_3.add(Dense(32,activation='relu'))

# Adding a dense output layer
model_3.add(Dense(1, activation='sigmoid'))

opt=Adam() #creates an instance of Adam optimizer
# Compile model
model_3.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Generating the summary of the model
model_3.summary()

# **Observtaion:**
# Summary of the vgg_model includes list of layers, output shape and number of parameters in each layers. The total number

history_3 = model_3.fit(train_datagen.flow(X_train_normalized,y_train,
                                       batch_size=batch_size,
                                       seed=42,
                                       shuffle=False),
                    epochs=epochs,
                    steps_per_epoch=X_train_normalized.shape[0] // batch_size,
                    validation_data=(X_val_normalized,y_val),
                    verbose=1)

# **Observation:**
# history_3 stores both the training and validation data at each epochs.

plt.plot(history_3.history['accuracy'])  #Display training accuracy
plt.plot(history_3.history['val_accuracy'])  #Display validation Accuracy
plt.title('Model Accuracy') #Set title
plt.ylabel('Accuracy') #set lable for y-axis
plt.xlabel('Epoch') #set label for X-axis
plt.legend(['Train', 'Validation'], loc='upper left') #Add legend
plt.show() #display the plot

model_3_train_perf = model_performance_classification(model_3, X_train_normalized,y_train) #calculates the performance metrics for training data

print("Train performance metrics")
print(model_3_train_perf)

plot_confusion_matrix(model_3,X_train_normalized,y_train) #print confusion matrix for training data

model_3_valid_perf = model_performance_classification(model_3, X_val_normalized,y_val)  #calculates performance metrics for validation data

print("Validation performance metrics")
print(model_3_valid_perf)

plot_confusion_matrix(model_3,X_val_normalized,y_val) #display confusion matrix for validation data

# **Observation:**
# The model has achieved 100% accuracy on the training set and 100% accuracy on the validation set.Its seems like model is

# Visualizing the predictions

plt.figure(figsize=(2,2))  #create a new figure
plt.imshow(X_val[4]) #display the image located at index 4
plt.show()  #dispaly the image
prediction = model_3.predict(X_val_normalized[4].reshape(1,200,200,3)) #use trained model to make prediction
predicted_label = prediction[0][0]>0.5  # Extract the predicted class label
print('Predicted Label:', 1 if predicted_label else 0)
# Fix indexing issue in y_val
true_label = y_val.iloc[4] #retrive actual true label for the image
print('True Label:', true_label)

plt.figure(figsize=(2,2))  #create a new figure
plt.imshow(X_val[10])#display the image located at index 10
plt.show()  #dispaly the image
prediction = model_3.predict(X_val_normalized[10].reshape(1,200,200,3)) #use trained model to make prediction
predicted_label = prediction[0][0]>0.5  # Extract the predicted class label
print('Predicted Label:', 1 if predicted_label else 0)
# Fix indexing issue in y_val
true_label = y_val.iloc[10] #retrive actual true label for the image
print('True Label:', true_label)

# **Observation:**
# For both image Predicted label and true label is matching.

# Model 4: (VGG-16 (Base + FFNN + Data Augmentation)

# - In most of the real-world case studies, it is challenging to acquire a large number of images and then train CNNs.
# - To overcome this problem, one approach we might consider is **Data Augmentation**.
# - CNNs have the property of **translational invariance**, which means they can recognise an object even if its appearanc
# -  Horizontal Flip (should be set to True/False)
# -  Vertical Flip (should be set to True/False)
# -  Height Shift (should be between 0 and 1)
# -  Width Shift (should be between 0 and 1)
# -  Rotation (should be between 0 and 180)
# -  Shear (should be between 0 and 1)
# -  Zoom (should be between 0 and 1) etc.
# Remember, **data augmentation should not be used in the validation/test data set**.

model_4 = Sequential()

# Adding the convolutional part of the VGG16 model
model_4.add(vgg_model)

# Flattening the output of the VGG16 model because it is from a convolutional layer
model_4.add(Flatten())

#Adding the Feed Forward neural network
model_4.add(Dense(256,activation='relu'))
model_4.add(Dropout(rate=0.4))
model_4.add(Dense(32,activation='relu'))

# Adding a dense output layer
model_4.add(Dense(1, activation='sigmoid'))

opt=Adam(learning_rate=0.001)
# Compile model
model_4.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Generating the summary of the model
model_4.summary()

# **Observtaion:**
# Summary of the vgg_model includes list of layers, output shape and number of parameters in each layers. The total number

# Applying data augmentation
train_datagen = ImageDataGenerator(
                              rotation_range=20,   #specifes a range within which rotation is applied
                              fill_mode='nearest',  #nearest fill them with the nearest value
                              width_shift_range=0.2, #image can be shifted horizontally with 20% of its total width
                              height_shift_range=0.2, #image can be randonly shifted vertically up to 20% of their total height
                              shear_range=0.3, #skew image along axis with the value of 0.3
                              zoom_range=0.4 #image can be zoomed out or in
                              )

history_4 = model_4.fit(train_datagen.flow(X_train_normalized,y_train,
                                       batch_size=batch_size,
                                       seed=42,
                                       shuffle=False),
                    epochs=epochs,
                    steps_per_epoch=X_train_normalized.shape[0] // batch_size,
                    validation_data=(X_val_normalized,y_val),
                    verbose=1)

# **Observation:**
# history_4 stores both the training and validation data at each epochs.

plt.plot(history_4.history['accuracy'])  #Display training accuracy
plt.plot(history_4.history['val_accuracy']) #display validation accuracy
plt.title('Model Accuracy') #set title
plt.ylabel('Accuracy') #set label for y-axis
plt.xlabel('Epoch') #set label for x-axis
plt.legend(['Train', 'Validation'], loc='upper left') #Add legend
plt.show() #display the plot

model_4_train_perf = model_performance_classification(model_4, X_train_normalized,y_train) #calculates performance metrics for training data

print("Train performance metrics")
print(model_4_train_perf)

plot_confusion_matrix(model_4,X_train_normalized,y_train) #display confusion matrix for training data

model_4_valid_perf = model_performance_classification(model_4, X_val_normalized,y_val)  #calculates performace metrics for validation data

print("Validation performance metrics")
print(model_4_valid_perf)

plot_confusion_matrix(model_4,X_val_normalized,y_val) #display confusion matrix for validation data

# **Observation:**
# The model has achieved 100% accuracy on the training set and 100% accuracy on the validation set.Its seems like model is

# Visualizing the predictions

plt.figure(figsize=(2,2)) #create a new figure
plt.imshow(X_val[4]) #display the image located at index 4
plt.show()  #dispaly the image
prediction = model_4.predict(X_val_normalized[4].reshape(1,200,200,3)) #use trained model to make prediction
predicted_label = prediction[0][0]>0.5  # Extract the predicted class label
print('Predicted Label:', 1 if predicted_label else 0)
# Fix indexing issue in y_val
true_label = y_val.iloc[4] #retrive actual true label for the image
print('True Label:', true_label)

plt.figure(figsize=(2,2))  #create a new figure
plt.imshow(X_val[10]) #display the image located at index 10
plt.show()   #dispaly the image
prediction = model_4.predict(X_val_normalized[10].reshape(1,200,200,3)) #use trained model to make prediction
predicted_label = prediction[0][0]>0.5  # Extract the predicted class label
print('Predicted Label:', 1 if predicted_label else 0)
# Fix indexing issue in y_val
true_label = y_val.iloc[10] #retrive actual true label for the image
print('True Label:', true_label)

# **Observation:**
# For both image Predicted label and true label is matching.

# **Model Performance Comparison and Final Model Selection**

# training performance comparison

models_train_comp_df = pd.concat(
    [
        model_1_train_perf.T,
        model_2_train_perf.T,
        model_3_train_perf.T,
        model_4_train_perf.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Simple Convolutional Neural Network (CNN)","VGG-16 (Base)","VGG-16 (Base+FFNN)","VGG-16 (Base+FFNN+Data Aug)"
]

# validation performance comparison
models_valid_comp_df = pd.concat(
    [
        model_1_valid_perf.T,
        model_2_valid_perf.T,
        model_3_valid_perf.T,
        model_4_valid_perf.T

    ],
    axis=1,
)
models_valid_comp_df.columns = [
 "Simple Convolutional Neural Network (CNN)","VGG-16 (Base)","VGG-16 (Base+FFNN)","VGG-16 (Base+FFNN+Data Aug)"
]

models_train_comp_df  #display metrics for each model

models_valid_comp_df  #display metrics for each model

models_train_comp_df - models_valid_comp_df  #calculates element wise subtraction between each dataframe and display

# Observation:
# All the model has acheived the same level of performance in the training set as well as validation set. We can choose an
# I am selecting model2 (VGG-16 (Base)) as our final model.

# Test Performance

model_test_perf = model_performance_classification(model_2, X_test_normalized,y_test)  #calculates the performance metrics on test data

model_test_perf  #display the metrics

plot_confusion_matrix(model_2, X_test_normalized,y_test) #Display confusion matrix for test data

# **Observation:**
# The model has achieved 100% accuracy on the test set as well.

# **Actionable Insights & Recommendations**

# **Actionable Insights**
# 1. Helmet Compliance Mapping:
# Analyze visual data to identify locations and time slots with consistently high helmet usage. These insights can highlig
# 2. Non-Compliance Heat Zones:
# Detect recurring patterns of violations based on shift timings, job types, or specific work locations. These patterns ca
# 3. Contractor-Level Risk Analysis:
# Correlate helmet violations with specific contractors, subcontractors, or teams to build a safety risk profile. This ena
# 4. Impact Assessment of Interventions:
# Monitor changes in compliance rates after introducing real-time alerts or training campaigns. Metrics such as repeat vio
# Business Recommendations
# 1. Prioritize Real-Time Monitoring at High-Risk Locations:
# Begin system rollout at sites with high incident potential—such as zones involving overhead machinery, scaffolding, or r
# 2. Integrate with HR and Safety Management Platforms:
# Link helmet compliance data with employee records and shift logs to support trend analysis, safety training, and predict
# 3. Introduce Safety Performance Indicators for Vendors:
# Establish helmet compliance KPIs for subcontractors, making adherence a component of contract renewals, bonuses, or pena
# 4. Automate Safety Reporting and Audit Readiness:
# Generate periodic compliance dashboards and downloadable reports to support internal audits and align with external regu
# 5. Install automated helmet detection at high-risk zones like machinery areas—to alert or restrict access for workers wi

# <font size=5 color='blue'>Power Ahead!</font>
# ___
