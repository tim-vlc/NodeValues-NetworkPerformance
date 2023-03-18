"""
File created on 13/01/2023 (dd/mm/yyyy)
@author: Tim Valencony
@author_email: tim.valencony@polytechnique.edu

FILE MAIN TITLE: TORCH LIBRARY

Purpose: create a file containing all the needed custom 
functions for our network generation file.
"""

# Import the needed libraries
import torch
from typing import List
from typing import Tuple
import pickle
import os
from filelock import Timeout, FileLock
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch.utils.data import random_split
import itertools
import numpy as np
import matplotlib.pyplot as plt
import math

def network_range(list_nv : list):
    """
    Computes the upper bound and lower bound of each node's values 
    during training.
    
    : param list_nv: list of list of arrays representing the node
    values of each feedforward over training.
    : returns: two lists of arrays representing the minimal and
    maximal values hit by the network over training for all nodes.
    : rtype: list of nupy.array, list of nupy.array
    """
    max_nv = []
    min_nv = []

    for i in range(len(list_nv[0])):
        layerI = [item[i] for item in list_nv]
        maxi   = np.amax(np.array(layerI), axis=0)
        mini   = np.amin(np.array(layerI), axis=0)
        max_nv.append(maxi)
        min_nv.append(mini)
    
    return min_nv, max_nv


def outside_range(min_nv : list, max_nv : list, sample_nv : list):
    """
    Computes when the node values, for a given sample, are outside
    the bounds of the trained network's range of node values.
    
    : param min_nv: list of arrays representing the minimal values
    hit by the network over training for all nodes.
    : param max_nv: list of arrays representing the maximal values
    hit by the network over training for all nodes.
    : param sample_nv: list of arrays representing the values hit
    by the network over the feedforward of a sample for all nodes.
    
    : returns: the number of node values outside of range + a list
    of arrays representing the node values, 0 if in range, value 
    of the node if outside (this was mainly used for debugging issues
    , but could surely be used / modified for further improvements).
    : rtype: int, list of numpy.array
    """
    num_out = []
    
    for arrv in sample_nv:
        counter = 0
        for minv, maxv, v in zip(min_nv, max_nv, arrv):
            if v < minv or v > maxv:
                counter += 1
        num_out.append(counter)
        
    return num_out

def network_probs(list_nv : list):
    """
    Generates the mean, variance, and standard deviation for 
    each node value, given a list of node values for different samples.
    
    : param list_nv: list of lists, the sublists contain the node 
    values for one test sample
    
    : returns: the mean, variance, and standard deviation per node
    : rtype: list, list, list
    """
    mean_nv = []
    stdv_nv = []
    
    num_samples = len(list_nv)
    num_nodes = len(list_nv[0])
    
    for i in range(num_nodes):
        meani = 0
        for j in range(num_samples):
            meani += list_nv[j][i]
        meani /= num_samples
        
        vari = 0
        for j in range(num_samples):
            vari += (list_nv[j][i] - meani)**2
        vari /= num_samples - 1
        
        stdv_nv.append(math.sqrt(vari))
        mean_nv.append(meani)
        
    return np.array(mean_nv), np.array(stdv_nv)

def dist_stdv(mean_nv, stdv_nv, test_nv):
    """
    Computes how far from the mean in terms of number of 
    standard deviations the node value for the test sample is.
    
    : param mean_nv: list of mean of node values obtained over training
    : param stdv_nv: list of standard deviation of node values obtained
    over training
    : test_nv: list of list of node values for each sample in the test
    set
    
    : returns: the list of "distances" for each node per sample
    : rtype: list
    """
    list_dist_nv = []
    for node_values in list(test_nv):
        res = []
        for nv, mean, stdv in zip(node_values, mean_nv, stdv_nv):
            if stdv > 0:
                res.append(abs(nv-mean)/stdv)
            else:
                res.append(0)
        list_dist_nv.append(res)
    return np.array(list_dist_nv)

def mean_dist_stdv(list_dist_nv):
    """
    Computes the mean of distances in terms of singular node values'
    standard deviation in the whole network per sample.
    
    : param list_dist_nv: the list of distances for each node per sample
    
    : returns: a list of mean distances per sample (takes all the 
    distances in the network for each sample and computes its mean).
    : rtype: list of floats
    """
    mean_dist = []
    for dist_nv in list_dist_nv:
        mean_dist.append(sum(dist_nv)/(len(dist_nv)))
    
    return mean_dist

def store_data(list_data:list, list_names:List[str], folder_name:str):
    """
    Stores the data we need for analysis and to draw 
    conclusions from our observations.

    : param list_data: list of python objects to be 
    serialized.
    : param list_names: names of the objects to be
    serilized.
    : param folder_name: name of the folder we want
    to store the data in.
    : returns: nothing, does everything in place.
    """
    # Serialize the data using pickle

    # Create the folder in which the files are going to be written
    try:
        cwd = os.getcwd()
        file_path = cwd + '/data/' + folder_name
        os.mkdir(file_path)
        print("Directory ", folder_name, " Created ") 
    except FileExistsError:
        print("Directory ", folder_name , " already exists")

    # Define a function to serialize the data
    def serialize(data, name):
        with open(os.path.join(file_path, name + ".pkl"), 'wb') as f:
            pickle.dump(data, f, protocol=2)

    for data, name in zip(list_data, list_names):
        serialize(data, name)

def writer(test_num,sampling_method,tot_mean,tot_out,gen_err,train_err,width,depth,state):
    cwd = os.getcwd()
    file_path = cwd + '/data/' + state + '/results_' + sampling_method + '.txt'

    lock = FileLock(file_path + '.lock', timeout=100)
    lock.acquire()
    try:
        open(file_path, "a").write(str(test_num) + ' ' + str(width) + ' ' + str(depth) + ' ' + str(tot_mean) + ' ' +
                                   str(tot_out) + ' ' + str(train_err) + ' ' + str(gen_err) + '\n')
    finally:
        lock.release(force=True)

#definition of a cost function
def mse(pred, truth):
    res = []
    for p, t in zip(pred, truth):
        res.append((p[0]-t[0])**2)
    return np.array(res)