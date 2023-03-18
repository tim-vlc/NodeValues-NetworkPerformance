from data_sample import sampling_set
from data_handling import store_data
from data_sample_hole import sampling_set_hole
import numpy as np

# Define the studied functions
def f1(x):
    return np.sin(x)
def f2(x):
    if x < 14:
        return 1+ 2*np.cos(x) - 3*np.cos(2*x) + np.cos(3*x)
    elif 14 <= x < 17:
        return np.sin(x*1.2) + 4.2
    elif 17 <= x:
        return np.sqrt(x) + 0.6
def f3(x, y):
    return x + y + np.sin(4*x*y)
def f4(x, y):
    """
    2D to 1D function which is composed of different functions
    """
    if -3 <= x < 1 and -1 <= y < 1:
        return x + y

    elif -3 <= x < -1:
        if 1 <= y <= 3:
            return -(x**2) - y**2 - x*y + 5
        elif -3 <= y < -1:
            return ((x+3)**2) * ((x+1)**2) * 3 * ((y+3)**2) * ((y+1)**2)
    
    elif -1 <= x < 1:
        if 1 <= y <= 3:
            return np.sin(x-2) / (y+2)
        elif -3 <= y < -1:
            return (-2)*x - (0.4*y) - (0.5*x*y) -1

    elif 1 <= x <= 3:
        if 1 <= y <= 3:
            return np.log(x*y)
        elif -1 <= y < 1:
            return x + y #x**2 + y**2
        elif -3 <= y < -1:
            return (x-2)**2 - (y+2)**2 - 1

# Generate train data
methods = ['REG', 'UNI', 'SOB']
f = f4
input_dim = 2
state = 'hole' # plain or hole

# Generate test data
for sampling_method in methods:
    folder_name = "f4/train_data_f4_" + state + "_"
    intervals = [((-3, 2),(1, 3)), ((-1, 1),(-1, 1)), ((-3, 2),(-3, -1))]
    n_sizes = [256]
    for n in n_sizes:
        new_fn = folder_name + str(43690)
        # x_train, y_train = sampling_set(f, sampling_method, n, input_dim)
        x_train, y_train = sampling_set_hole(f, sampling_method, n, input_dim, intervals)
        print(len(x_train))
        store_data([x_train, y_train], [sampling_method + '_x_train', sampling_method + '_y_train'], new_fn)

# Generate test data
# n_sizes = [150]
# #intervals = [(9.5, 21.5)]
# for n in n_sizes:
#     folder_name = "f4/test_data_f4_plain_" + str(n**2)
#     x_test, y_test = sampling_set(f, 'REG', n, input_dim) #sampling_set_hole(f, 'REG', n, input_dim, intervals)
#     print(len(x_test))
#     store_data([x_test, y_test], ['x_test',  'y_test'], folder_name)