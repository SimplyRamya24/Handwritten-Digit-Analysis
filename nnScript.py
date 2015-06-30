# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import random
import pickle

def initializeWeights(n_in,n_out):
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



def sigmoid(z):
    
    return  1/(1+np.exp(np.multiply(-1,z)))

def get_dummies(label):
    rows = label.shape[0];
    rowsIndex=np.arange(rows,dtype="int")
    # Below line can be hardcoded in our case 
    oneKLabel = np.zeros((rows,10))
    #oneKLabel = np.zeros((rows,np.max(label)+1))
    oneKLabel[rowsIndex,label.astype(int)]=1
    return oneKLabel


def featureReduction(data):
    deleteIndices = [];
    #Tweaks added for optimizing
    for i in range(0,data.shape[1]):
        if ((data[:,i] - data[0,i]) == 0).all():
            deleteIndices += [i];
    #data_temp = np.delete(data,deleteIndices,1)
    return deleteIndices
    
    
def preprocess():
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary    
    A = np.zeros((0,784))
    Alabel = []

    test_data = np.zeros((0,784))
    test_label = []

    # stacking training and testing data 
    for i in range(10):
        train = "train" + str(i)
        trainData = mat.get(train)
        A = np.concatenate((A,trainData),0)
        Alabel=np.concatenate((Alabel,np.ones(trainData.shape[0])*i),0);

        test = "test" + str(i)
        test_data = np.concatenate((test_data,mat.get(test)),0)
        test_label = np.concatenate((test_label,np.ones(mat.get(test).shape[0])*i),0)
        
    # normalizing trainig (validation) and testing data  
    A = np.double(A)
    test_data = np.double(test_data)
        
    C = np.where(A>0)
    A[C] = A[C]/255.0

    D = np.where(test_data>0)
    test_data[D] = test_data[D]/255.0


    # spliting train_data into train_data and validation_data
    train_data = np.zeros((0,784))
    train_label = np.zeros((50000))

    validation_data = np.zeros((0,784))
    validation_label = np.zeros((10000))
    
    # Random samples
    s = random.sample(range(A.shape[0]),A.shape[0])
    
    # Reduce features for the dataset using train
    deleteIndices = featureReduction(A)
    
    # Get Reduced train and test
    A = np.delete(A,deleteIndices,1)
    test_data = np.delete(test_data, deleteIndices,1)
    
    # Separate train and validation    
    train_data = A[s[0:50000],:]
    train_label = Alabel[s[0:50000]]; 
    
    
    validation_data =A[s[50000:60000],:]
    validation_label = Alabel[s[50000:60000]];       
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    

def nnObjFunction(params, *args):
     
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args 
    
    # one of k encoding
    training_label = get_dummies(np.array(training_label))

    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    

    # Feedforward
    # Add bias
    training_data = np.column_stack((training_data,np.ones(training_data.shape[0])))
    # Feed to hidden
    zj_array_n = sigmoid(np.dot(training_data,w1.T))
    # Add bias
    zj_array_n = np.column_stack((zj_array_n,np.ones(zj_array_n.shape[0])))
    # Feed to output
    ol_array_n = sigmoid(np.dot(zj_array_n,w2.T))
    
    # Back propogation
    delta_l = ol_array_n - training_label
    
    grad_w2 = np.dot(delta_l.T,zj_array_n)
    grad_w1 = np.dot(((1-zj_array_n)*zj_array_n* (np.dot(delta_l,w2))).T,training_data)  
    
    # Remove zero row
    grad_w1 = np.delete(grad_w1, n_hidden,0)
    
    num_samples = training_data.shape[0]

    # obj_grad  
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = obj_grad/num_samples

    # obj_val
    obj_val_part1 = np.sum(-1*(training_label*np.log(ol_array_n)+(1-training_label)*np.log(1-ol_array_n)))
    obj_val_part1 = obj_val_part1/num_samples
    obj_val_part2 = (lambdaval/(2*num_samples))* ( np.sum(np.square(w1)) + np.sum(np.square(w2)))    
    obj_val = obj_val_part1 + obj_val_part2
    
    return (obj_val,obj_grad)

def nnPredict(w1,w2,data):    
    # Add bias
    data = np.column_stack((data,np.ones(data.shape[0])))
    zj_array_n = sigmoid(np.dot(data,w1.T))
    # Add bias
    zj_array_n = np.column_stack((zj_array_n,np.ones(zj_array_n.shape[0])))
    # Feed to output
    ol_array_n = sigmoid(np.dot(zj_array_n,w2.T))
    
    # Return indices of max as labels
    labels = np.argmax(ol_array_n,axis=1)
    return labels
    

"""**************Neural Network Script Starts here********************************"""

tic = time.time()
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

 #set the number of nodes in hidden unit (not including bias unit)
n_hidden = 80;
				   
# set the number of nodes in output unit
n_class = 10;				   


# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.4;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

toc = time.time()-tic
print('\n Time taken in seconds: ' + str(toc))


pickle.dump((n_hidden,w1,w2,lambdaval),open('params.pickle','wb'))

written = pickle.load(open('params.pickle','rb'))
print written


