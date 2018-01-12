# E4040 A-Convolutional-Neural-Network-Model-for-Multi-digit-Number-Recognition-from-Street-View-Imagery

# Overview of all files
- train_and_test_in_jupyter.ipynb: Main code, all functions run in this notebook file, and all results are printed here
-  train_forJupyter.py: This script defines the method to train our model
- test_forJupyter.py: This script defines the method to test the trained model
- model.py: This script defines the structure of the CNN
- evaluator.py: This script defines the method to evaluate the prediction accuracy of the model
- batch.py: This script defines the method to split the data set into batches
- data.py: This script defines the method which follow a certain protocol to read from and write to the .json file

# Description of our project
A deep convolutional neural network has been designed and implemented to recognize multi-digit number from street view imagery in this project. The overall structure of the CNN designed in this paper is derived from the model discussed in the paper published by Ian J. Goodfellow, et al. In order to optimize the accuracy of testing result, hyper-parameters such as size of convolutional layer have been tuned.  To further discuss the performance of our CNN model, it is expected to train with more data to see whether the final accuracies can reach ones given in Goodfellow’s paper with the same dataset.

