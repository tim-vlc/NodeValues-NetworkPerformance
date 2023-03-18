import numpy as np
import torch
from scipy.stats import qmc
import math

# State the ground truth function


def closest_power_of_two(n):
    power = 1
    while power < n:
        power *= 2
    if abs(power - n) < abs(power//2 - n):
        return int(math.log2(power))
    else:
        return int(math.log2(power // 2))

def sampling_set(f, sampling_method,n, input_dim):
    num_samples = n**2
    
    if sampling_method == 'REG':
        if input_dim == 1:
            return data_REG_1D(num_samples, f)
        return data_REG_2D(n, f)
    elif sampling_method == 'UNI':
        if input_dim == 1:
            return data_UNI_1D(num_samples, f)
        return data_UNI_2D(num_samples,f)
    elif sampling_method == 'SOB':
        if input_dim == 1:
            return data_SOB_1D(num_samples,f)
        return data_SOB_2D(num_samples,f)

def data_REG_2D(n,f):
    X = np.linspace(-3, 3, n)
    Y = np.linspace(-3, 3, n)
    XY = []
    Z = []
    for x in X:
        for y in Y:
            XY.append(np.array([x, y]))
            z = f(x, y)
            Z.append(z)
    XY = np.array(XY)
    x_train = torch.from_numpy(XY)
    y_train = torch.from_numpy(np.array(Z))

    return x_train, y_train

def data_UNI_2D(num_samples,f):
    x_train = np.random.uniform(-3, 3, size=(num_samples, 2))
    y_train = [f(sample[0], sample[1]) for sample in x_train]
    x_train = torch.from_numpy(np.array(x_train))
    y_train = torch.from_numpy(np.array(y_train))

    return x_train, y_train

def data_SOB_2D(num_samples,f):
    ndim = 2
    m2 = closest_power_of_two(num_samples)

    # Generate a Sobol sequence of points
    sampler = qmc.Sobol(d=ndim, scramble=False)
    x_train = sampler.random_base2(m=m2)
    # Scale the points to be within the range [-3, 3]
    x_train = 6 * x_train - 3
    y_train = [f(sample[0], sample[1]) for sample in x_train]
    x_train = torch.from_numpy(np.array(x_train))
    y_train = torch.from_numpy(np.array(y_train))

    return x_train, y_train

def data_REG_1D(num_samples,f):
    X = np.linspace(-3, 3, num_samples).reshape(num_samples, 1)
    Y = f(X)
    x_train = torch.from_numpy(X)
    y_train = torch.from_numpy(Y)

    return x_train, y_train

def data_UNI_1D(num_samples,f):
    X = np.random.uniform(low=-3, high=3, size=num_samples).reshape(num_samples, 1)
    Y = f(X)
    x_train = torch.from_numpy(X)
    y_train = torch.from_numpy(Y)

    return x_train, y_train

def data_SOB_1D(num_samples,f):
    ndim = 1
    m2 = closest_power_of_two(num_samples)

    # Generate a Sobol sequence of points
    sampler = qmc.Sobol(d=ndim, scramble=False)
    x_train = sampler.random_base2(m=m2)
    # Scale the points to be within the range [-3, 3]
    x_train = 6 * x_train - 3
    y_train = [f(x) for x in x_train]
    x_train = torch.from_numpy(np.array(x_train))
    y_train = torch.from_numpy(np.array(y_train))

    return x_train, y_train