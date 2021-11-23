import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import yaml

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
class logistic_regression_mnist:
    def __init__(self):
        print("Initialization")
        lr_param_file = open("../params/config.yaml", 'r')
        lr_params = yaml.load(lr_param_file)
        self.num_iter = lr_params["num_iter"]
        self.learning_rate = lr_params["learning_rate"]
        self.num_column = lr_params["num_column"]
        self.num_row = lr_params["num_row"]
        self.total_mnist_dataset = {}
        self.final_model_params = []
        self.final_objective_values = []
        self.training_bias = 0 
        self.objective_func_minimizers = []
        self.final_loss_value = lr_params["final_loss"]
        self.activation_value = lr_params["activation_value"]
        self.logistic_func_weights = np.random.randn(1,self.num_row)
    
    def sigmoid(self, fx):
        return 1/(1 + np.exp(-fx))
    
    def calc_logistic_function(self, logistic_func_weights, X, training_bias):
        return np.dot(logistic_func_weights,X) + training_bias

    def train(self): 
        print("Please wait for the training process to be done!")                 
        data = self.total_mnist_dataset['mnist'][0] 
        label = self.total_mnist_dataset['mnist'][1]
        print("Training.........................")
        for i in range(1, self.num_iter + 1):
            # Calculating logistic function
            logistic_function_value = self.calc_logistic_function(self.logistic_func_weights, data, self.training_bias)
            sigmoid_final_value = self.sigmoid(logistic_function_value) 

            # Calculating objective function
            obj_func_value = 1/self.num_column*(-1*(np.sum(label*np.log(sigmoid_final_value) + (1-label)*np.log(1-sigmoid_final_value))))
            self.objective_func_minimizers.append(obj_func_value)

            # Calculating gradient decent
            self.logistic_func_weights = self.logistic_func_weights - self.learning_rate*(1/self.num_column * np.dot(sigmoid_final_value - label,data.T))
            self.training_bias = self.training_bias - self.learning_rate*(1/self.num_column * np.sum(sigmoid_final_value - label))
            if (obj_func_value) < 0.08:
                break

        self.final_objective_values.append(self.objective_func_minimizers)  
        self.final_model_params.append([self.logistic_func_weights,self.training_bias])
        print("Done training!")
    
    def get_dataset(self):
        data = pd.read_csv('../mnist_data/mnist_dataset.csv',header=None)
        datasets = [data]
        for i in range(self.num_column):
            if datasets[0].at[i,self.num_row] == 0:
                datasets[0].at[i,self.num_row] = 1
            else:
                datasets[0].at[i,self.num_row] = 0
          
        data_x = datasets[0].iloc[:,:self.num_row]
        data_x = data_x.T
        label = datasets[0].iloc[:,-1]
        label = np.array([label])
        self.total_mnist_dataset['mnist'] = [data_x, label]
        return self.total_mnist_dataset

    def get_accuracy(self):
        data = self.total_mnist_dataset['mnist'][0]
        label = self.total_mnist_dataset['mnist'][1]
        logistic_func_weights = self.final_model_params[0][0]
        self.training_bias = self.final_model_params[0][1]
        true_positives = 0 
        for i in range(self.num_column):
            fx = np.dot(logistic_func_weights,data.T.iloc[i,:]) + self.training_bias
            sigmoid_final_value = self.sigmoid(fx)
            if np.logical_and(sigmoid_final_value >= self.activation_value,label.T[i,0] == 1):
                true_positives += 1
            if np.logical_and(sigmoid_final_value < self.activation_value,label.T[i,0] == 0):
                true_positives += 1
        print('The Model Accuracy: ',100.0*(true_positives/self.num_column))

    def plot_results(self):
        plt.plot(self.final_objective_values[0], LineWidth=2)
        plt.xlabel("Interation #")
        plt.ylabel("Loss Value")
        plt.grid()
        plt.show()