import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from modules import logistic_regression_mnist

"""
(Coding) Download the benchmark dataset MNIST from http://yann.lecun.com/exdb/mnist/.
Implement multiclass logistic regression and try it on MNIST.
Comments: MNIST is a standard dataset for machine learning and also deep learning. It's
good to try it on one layer neural networks (i.e., logistic regression) before multilayer neural
networks. Downloading the dataset from other places in preprocessed format is allowed, but
practicing how to read the dataset prepares you for other new datasets you may be
interested in. Also, it is recommended to try different initializations and learning rates to get
a sense about how to tune the hyperparameters (remember to create and use validation
dataset!).
"""

if __name__ == "__main__":
      
      # Create an object from my mnist modules
      lr_mnist = logistic_regression_mnist()

      # Parse the mnist dataset (data.csv)
      lr_mnist.get_dataset()

      # Start training parameters
      lr_mnist.train()

      # Calculate accuracy
      lr_mnist.get_accuracy()

      # Plot results
      lr_mnist.plot_results()

