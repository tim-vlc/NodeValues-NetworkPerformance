import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import matplotlib
import os
import matplotlib.tri as tri

def make_3D_plot(X, Y, Z, file_name, sampling_method):
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z)
    ax = fig.gca(projection='3d')
    graph = ax.plot_trisurf(X, Y, Z, cmap=matplotlib.cm.coolwarm)
    fig.colorbar(graph, shrink=0.5, aspect=15)
    ax.set_title("Ground truth function $f_3$") #"Estimation of function $f_4$ with sampling method " + sampling_method)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    fig.savefig(file_name + '.png', dpi=200)

def make_contour_plot(X, Y, Z, file_name, status, sampling_method):

    # prepare the interpolator
    triang = tri.Triangulation(X, Y)
    interpolator = tri.LinearTriInterpolator(triang, Z)

    # do the interpolation
    xi = yi = np.linspace(-2, 2, 96)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = interpolator(Xi, Yi)

    # pcolormesh needs the pixel edges for x and y
    # and with default flat shading, Z needs to be evaluated at the pixel center
    plot = plt.pcolormesh(xi, yi, Zi, cmap='viridis', shading='flat')
    
    num_levels = 6
    if status == 'err':
        num_levels = 10
    elif status == 'out':
        num_levels = 4

    # contour needs the centers
    cset = plt.contour(Xi, Yi, Zi, num_levels, cmap='Set2')
    plt.clabel(cset, inline=True)

    colorbar = plt.colorbar(plot)

    # plt.scatter(X, Y, c=Z, cmap='viridis')
    # colorbar = plt.colorbar()

    if status == 'out':
        colorbar.ax.set_ylabel(r"$\mathcal{O}_j$", fontsize=13)
        plt.title(r"Plot of samples $x_j = (x^1_j, x^2_j)$ compared to $\mathcal{O}_j$, with sampling method " + sampling_method, fontsize=14)
    elif status == 'err':
        colorbar.ax.set_ylabel(r"MSE", fontsize=13)
        plt.title(r"Plot of samples $x_j = (x^1_j, x^2_j)$ compared to MSE, with sampling method " + sampling_method, fontsize=14)
    else:
        colorbar.ax.set_ylabel(r"$\Delta_j$", fontsize=13)
        plt.title(r"Plot of samples $x_j = (x^1_j, x^2_j)$ compared to $\Delta_j$, with sampling method " + sampling_method, fontsize=14)
    plt.xlabel('$x^1_j$', fontsize=13)
    plt.ylabel('$x^2_j$', fontsize=13)
    plt.savefig(file_name + '.png', dpi=200)

# Define a function to deserialize the data
def deserialize(name, folder_name):
    with open(os.path.join(folder_name, name + ".pkl"), 'rb') as f:
        data = pickle.load(f, encoding='latin1', fix_imports=True)
    return data

width = 10
depth = 3
test_num = 5
sampling_method = 'REG'
state = 'plain' # Either hole or plain

train_size = 24649 if state == 'hole' else 16384
test_size = 9216
train_path = 'data/samples/train_data_f3_' + state + '_' + str(train_size)
test_path = 'data/samples/test_data_f3_plain_' + str(test_size) + '/'

# Training and Testing samples
x_train, y_train = deserialize(sampling_method + "_x_train", train_path), deserialize(sampling_method + "_y_train", train_path)
x_test, y_test = deserialize("x_test", test_path), deserialize("y_test", test_path)
train_size = len(x_train)
test_size = len(x_test)

path = "data/serialized_" + state + "/serialized_data_" + sampling_method

# Deserialize the data using pickle
folder_name = path + f"_{width}_{depth}_{test_num}"
list_names = ["mean_dist", "num_out", "output_test", "gensample_err"]
list_data = []
for name in list_names:
    list_data.append(deserialize(name, folder_name))
mean_dist = list_data[0]
num_out = list_data[1]
output_test = list_data[2].detach().numpy().reshape(len(y_test))
gen_err = list_data[3]
y_test = y_test.detach().numpy()
contour_path = "images/Contour_" + str(state) + "_" + str(sampling_method) + "_" + str(width) + "_" + str(depth) + "_" + str(test_num)
dimensional_path = "images/3D_" + str(state) + "_" + str(sampling_method) + "_" + str(width) + "_" + str(depth) + "_" + str(test_num)
truth_path = "images/3D_ground_truth"

# make_contour_plot(x_test[:, 0], x_test[:, 1], num_out, contour_path + "_out", "out", sampling_method)
# plt.clf()
# make_contour_plot(x_test[:, 0], x_test[:, 1], mean_dist, contour_path + "_dist", "dist", sampling_method)
# plt.clf()
# make_contour_plot(x_test[:, 0], x_test[:, 1], gen_err, contour_path + "_err", "err", sampling_method)
# plt.clf()

# make_3D_plot(x_test[:, 0], x_test[:, 1], output_test, dimensional_path, sampling_method)

make_3D_plot(x_test[:, 0], x_test[:, 1], y_test, truth_path, sampling_method)


# # Define the fraction you want to select
# fraction = 0.1984

# # Determine the number of rows to select
# num_rows = int(len(x_test) * fraction)

# # Set a random seed for reproducibility
# np.random.seed(42)

# # Randomly select the rows
# selected_rows = np.random.choice(np.arange(len(x_test)), num_rows, replace=False)

# # Sort the selected rows for consistent order
# selected_rows = np.sort(selected_rows)

# # Select the rows from the original array and create a new array
# x_train = x_train[selected_rows]
# y_train = y_train[selected_rows]

# # prepare the interpolator
# triang = tri.Triangulation(x_train[:, 0], x_train[:, 1])
# interpolator = tri.LinearTriInterpolator(triang, y_train)

# # do the interpolation
# xi = yi = np.linspace(-3, 3, 114)
# Xi, Yi = np.meshgrid(xi, yi)
# Zi = interpolator(Xi, Yi)

# # Plot the surface
# make_3D_plot(Xi, Yi, Zi, "hole_comp2D_plot")