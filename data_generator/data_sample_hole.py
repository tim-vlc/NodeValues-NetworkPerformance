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

def sampling_set_hole(f, sampling_method,n, input_dim, intervals):
    num_samples = n**2
    
    if sampling_method == 'REG':
        if input_dim == 1:
            return data_REG_1D(num_samples, f, intervals)
        return data_REG_2D(n, f, intervals)
    elif sampling_method == 'UNI':
        if input_dim == 1:
            return data_UNI_1D(num_samples, f,intervals)
        return data_UNI_2D(num_samples,f,intervals)
    elif sampling_method == 'SOB':
        if input_dim == 1:
            return data_SOB_1D(num_samples,f,intervals)
        return data_SOB_2D(num_samples,f,intervals)

def data_REG_2D(n,f, intervals):
    X = np.linspace(-3, 3, n)
    Y = np.linspace(-3, 3, n)
    x_train = []
    for x in X:
        for y in Y:
            x_train.append([x, y])
    x_train = np.array(x_train)

    res = np.array([[1, 2]])
    for ((a, b), (c, d)) in intervals:
        mask = (a <= x_train[:, 0]) & (x_train[:, 0] <= b) & (c <= x_train[:, 1]) & (x_train[:, 1] <= d)
        res = np.concatenate((res, x_train[mask]))

    x_train = res[1:]
    y_train = [f(sample[0], sample[1]) for sample in x_train]
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(np.array(y_train))

    return x_train, y_train

def data_UNI_2D(num_samples,f,intervals):
    x_train = np.random.uniform(-3, 3, size=(num_samples, 2))

    res = np.array([[1, 2]])
    for ((a, b), (c, d)) in intervals:
        mask = (a <= x_train[:, 0]) & (x_train[:, 0] <= b) & (c <= x_train[:, 1]) & (x_train[:, 1] <= d)
        res = np.concatenate((res, x_train[mask]))
    x_train = res[1:]
    
    y_train = [f(sample[0], sample[1]) for sample in x_train]
    x_train = torch.from_numpy(np.array(x_train))
    y_train = torch.from_numpy(np.array(y_train))

    return x_train, y_train

def data_SOB_2D(num_samples,f,intervals):
    ndim = 2
    m2 = closest_power_of_two(num_samples)

    # Generate a Sobol sequence of points
    sampler = qmc.Sobol(d=ndim, scramble=False)
    x_train = sampler.random_base2(m=m2)
    # Scale the points to be within the range [-3, 3]
    x_train = 6 * x_train - 3

    res = np.array([[1, 2]])
    for ((a, b), (c, d)) in intervals:
        mask = (a <= x_train[:, 0]) & (x_train[:, 0] <= b) & (c <= x_train[:, 1]) & (x_train[:, 1] <= d)
        res = np.concatenate((res, x_train[mask]))
    x_train = res[1:]

    y_train = [f(sample[0], sample[1]) for sample in x_train]
    x_train = torch.from_numpy(np.array(x_train))
    y_train = torch.from_numpy(np.array(y_train))

    return x_train, y_train

def data_REG_1D(num_samples,f,intervals):
    # Calculate the total length of all intervals
    total_length = sum([interval[1] - interval[0] for interval in intervals])

    # Calculate the number of samples for each interval proportional to its length
    num_samples_per_interval = [int(round(num_samples * (interval[1] - interval[0]) / total_length)) for interval in intervals]

    # Create the samples for each interval using np.linspace
    X = []
    for interval, num_samples in zip(intervals, num_samples_per_interval):
        X.extend(np.linspace(interval[0], interval[1], num_samples).reshape(num_samples, 1))
    Y = [f(x) for x in X]
    x_train = torch.from_numpy(np.array(X))
    y_train = torch.from_numpy(np.array(Y))

    return x_train, y_train

def data_UNI_1D(num_samples,f,intervals):
    # Calculate the total length of all intervals
    total_length = sum([interval[1] - interval[0] for interval in intervals])

    # Calculate the number of samples for each interval proportional to its length
    num_samples_per_interval = [int(round(num_samples * (interval[1] - interval[0]) / total_length)) for interval in intervals]

    # Create the samples for each interval using np.linspace
    X = []
    for interval, num_samples in zip(intervals, num_samples_per_interval):
        X.extend(np.random.uniform(low=interval[0], high=interval[1], size=num_samples).reshape(num_samples, 1))
    Y = [f(x) for x in X]
    x_train = torch.from_numpy(np.array(X))
    y_train = torch.from_numpy(np.array(Y))

    return x_train, y_train

def data_SOB_1D(num_samples,f,intervals):
    # Calculate the total length of all intervals
    total_length = sum([interval[1] - interval[0] for interval in intervals])

    # Calculate the number of samples for each interval proportional to its length
    num_samples_per_interval = [int(round(num_samples * (interval[1] - interval[0]) / total_length)) for interval in intervals]

    # Generate a Sobol sequence of points for each interval
    x_train = []
    y_train = []
    for interval, num_samples in zip(intervals, num_samples_per_interval):
        # Generate a Sobol sequence of points within the interval
        sampler = qmc.Sobol(d=1, scramble=False)
        m2 = closest_power_of_two(num_samples)
        x_interval = sampler.random_base2(m=m2)
        x_interval = (interval[1] - interval[0]) * x_interval + interval[0]
        y_interval = [f(x) for x in x_interval]
        x_train.extend(x_interval)
        y_train.extend(y_interval)

    x_train = torch.from_numpy(np.array(x_train))
    y_train = torch.from_numpy(np.array(y_train))

    return x_train, y_train