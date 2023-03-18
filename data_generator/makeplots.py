import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def make_3D_plot(X, Y, Z, file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    fig.savefig(file_name + '.png')

# Define a function to deserialize the data
def deserialize(name, folder_name):
    with open(os.path.join(folder_name, name + ".pkl"), 'rb') as f:
        data = pickle.load(f, encoding='latin1', fix_imports=True)
    return data

sampling_method = 'REG'
state = 'hole' # Either hole or plain
train_size = 43690
test_size = 22500
train_path = 'f4/train_data_f4_' + state + '_' + str(train_size)
test_path = 'f4/test_data_f4_plain_' + str(test_size) + '/'

# Training and Testing samples
x_train, y_train = deserialize(sampling_method + "_x_train", train_path), deserialize(sampling_method + "_y_train", train_path)
x_test, y_test = deserialize("x_test", test_path), deserialize("y_test", test_path)
train_size = len(x_train)
test_size = len(x_test)

make_3D_plot(x_train[:, 0], x_train[:, 1], np.reshape(y_train, (train_size, 1)), "hole_comp2D_plot")