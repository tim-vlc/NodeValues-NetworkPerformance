"""
File created on 13/01/2023 (dd/mm/yyyy)
@author: Tim Valencony
@author_email: tim.valencony@polytechnique.edu

FILE MAIN TITLE: TEST AUTOMATION OF NETWORKS

Purpose: File that automatically tests different network topologies
and functions so as to ascertain a relationship between the node 
values and the prediction power.
"""
#import ann_netgen as netgen
import numpy as np
import torch
import os
import pickle
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from torchnet import train, Network_architecture
from data_sample import closest_power_of_two, sampling_set, data_UNI

# Define a test function
def test(width,depth,lr,µ,input_dim,output_dim,samples,train_size,test_size,sampling_method,max_epochs,activationFunction,test_num,state):
    model = Network_architecture(lr, µ, width, depth)
    _,_,_,_,_ = train(model,test_num,samples,input_dim,output_dim,sampling_method,max_epochs,activationFunction,train_size,test_size,state)
# Define a function to deserialize the data
def deserialize(name, folder_name):
    with open(os.path.join(folder_name, name + ".pkl"), 'rb') as f:
        data = pickle.load(f, encoding='latin1', fix_imports=True)
    return data

################# CHOOSE SAMPLING METHOD #################
sampling_method = 'SOB' # Either REG, UNI, or SOB for regular (grid), random (uniform), and Sobol
state = 'hole' # Either hole or plain
train_size = 1024
test_size = 529
train_path = 'data/samples/train_data_f1_' + state + '_' + str(train_size)
test_path = 'data/samples/test_data_f1_plain_' + str(test_size) + '/'

# Training and Testing samples
x_train, y_train = deserialize(sampling_method + "_x_train", train_path), deserialize(sampling_method + "_y_train", train_path)
x_test, y_test = deserialize("x_test", test_path), deserialize("y_test", test_path)
train_size = len(x_train)
test_size = len(x_test)
print(train_size, test_size)

## Create the network
lr = 0.01
µ = 1.0e-04
input_dim = 1
output_dim = 1
samples = [x_train, y_train, x_test, y_test]
max_epochs = 1000
activationFunction = nn.Sigmoid() #nn.ReLU or nn.sigmoid()

archs = [(6, 1), (7, 1), (12, 1), (17, 1), (22, 1), (27, 1), (32, 1),
    (6, 2), (8, 2), (10, 2), (12, 2), (14, 2), (16, 2),
    (6, 3), (8, 3), (10, 3)]

test(6,2,lr,µ,input_dim,output_dim,samples,train_size,test_size,sampling_method,max_epochs,activationFunction,1,state)
# for width, depth in archs:
#     for test_num in range(1, 18):
#         test(width,depth,lr,µ,input_dim,output_dim,samples,train_size,test_size,sampling_method,max_epochs,activationFunction,test_num,state)