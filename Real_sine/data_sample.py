import numpy as np
import torch
import math
#from scipy.stats import qmc

# State the ground truth function
def sine(x):
    return np.sin(x)

def closest_power_of_two(n):
    power = 1
    while power < n:
        power *= 2
    if abs(power - n) < abs(power//2 - n):
        return int(math.log2(power))
    else:
        return int(math.log2(power // 2))

def sampling_set(sampling_method,num_samples):
    if sampling_method == 'REG':
        return data_REG(num_samples)
    elif sampling_method == 'UNI':
        return data_UNI(num_samples)
    elif sampling_method == 'SOB':
        return data_SOB(num_samples)

def data_REG(num_samples):
    X = np.linspace(-3, 3, num_samples).reshape(num_samples, 1)
    Y = sine(X)
    x_train = torch.from_numpy(X)
    y_train = torch.from_numpy(Y)

    return x_train, y_train

def data_UNI(num_samples):
    X = np.random.uniform(low=-3, high=3, size=num_samples).reshape(num_samples, 1)
    Y = sine(X)
    x_train = torch.from_numpy(X)
    y_train = torch.from_numpy(Y)

    return x_train, y_train

# def data_SOB(num_samples):
#     ndim = 1
#     m2 = closest_power_of_two(num_samples)

#     # Generate a Sobol sequence of points
#     sampler = qmc.Sobol(d=ndim, scramble=False)
#     x_train = sampler.random_base2(m=m2)
#     # Scale the points to be within the range [-3, 3]
#     x_train = 6 * x_train - 3
#     y_train = sine(x_train)
#     x_train = torch.from_numpy(np.array(x_train))
#     y_train = torch.from_numpy(np.array(y_train))

#     return x_train, y_train