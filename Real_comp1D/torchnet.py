import torch
from torch import nn, optim
import numpy as np
from data_handling import network_probs, dist_stdv, mean_dist_stdv, writer, store_data, network_range, outside_range


class Network_architecture(nn.Module):
    def __init__(self, learning_rate, regression_param, width, depth):
        super().__init__()
        self.learning_rate = learning_rate
        self.regression_param = regression_param
        self.width = int(width)
        self.depth = int(depth)
        self.layers = nn.ModuleList()
        self.node_values = []

    def forward(self, x, activationFunction):

        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = activationFunction(x)
            #nv_list.append(x.tolist())
        
        # Output layer
        x = self.layers[-1](x)

        #self.node_values += add_layers(nv_list)
        
        return x

def get_node_values(module, input, output, model):
    model.node_values.append(output.detach().numpy())

def reformat_nv(collected_nv, sample_size, max_epochs, depth):
    node_values = [[] for i in range((sample_size)*max_epochs)]
    for i in range(max_epochs):
        for h in range(sample_size):
            for j in range(depth):
                node_values[(i*sample_size)+h] += list(collected_nv[j+(i*depth)][h])
    return node_values

def add_layers(nv_list:list):
    node_values = []
    if len(nv_list) != 0:
        for i in range(len(nv_list[0])):
            add = []
            for layer in nv_list:
                add += layer[i]
            node_values.append(add)
    return node_values

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)

def network_model(model, layer_sizes, activationFunction, batch_norm=True, Xavier_init=True):
    if(batch_norm):
        model.layers.append(nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.BatchNorm1d(num_features=layer_sizes[1]),
            activationFunction()))
        for i in range(1, len(layer_sizes) - 2):
            model.layers.append(nn.Sequential(
                model, nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                nn.BatchNorm1d(num_features=layer_sizes[i + 1]),
                activationFunction()))
    else:
        model.layers.append(nn.Linear(layer_sizes[0], layer_sizes[1]))
        for i in range(1, len(layer_sizes) - 2):
            model.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
    model.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    if(Xavier_init):
        model.apply(init_weights)

    return model

def train(model,test_num,samples,input_dim,output_dim,sampling_method,max_epochs,activationFunction,train_size,test_size,state):
    x_train, y_train, x_test, y_test = samples[0], samples[1], samples[2], samples[3]

    layer_sizes = [input_dim]
    for i in range(model.depth):
        layer_sizes.append(model.width)
    layer_sizes.append(output_dim)

    model = network_model(model, layer_sizes, activationFunction, batch_norm=False)

    for i in range(0, len(model.layers)-1):
        model.layers[i].register_forward_hook(lambda m, i, o: get_node_values(m, i, o, model))

    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model.learning_rate, weight_decay=model.regression_param)

    for e in range(max_epochs):
        optimizer.zero_grad()
        output = model.forward(x_train.float(), activationFunction)
        loss = loss_func(output, y_train.float())
        loss.backward()
        optimizer.step()

    train_nv = np.array(reformat_nv(model.node_values,train_size,max_epochs,model.depth))
    training_error = loss.item()
    model.node_values = []

    output_test = model(x_test.float(), activationFunction)
    generalization_error = loss_func(output_test, y_test.float()).item()

    test_nv = np.array(reformat_nv(model.node_values,test_size,1,model.depth))
    model.node_values = []

    min_nv, max_nv = network_range(train_nv)
    num_out = outside_range(min_nv, max_nv, test_nv)
    mean_nv, stdv_nv = network_probs(train_nv)
    list_dist_nv = dist_stdv(mean_nv, stdv_nv, test_nv)
    mean_dist = mean_dist_stdv(list_dist_nv)
    tot_mean = sum(mean_dist) / len(mean_dist)
    tot_out = sum(num_out) / len(num_out)

    folder_name = 'serialized_' + state + "/serialized_data_" + sampling_method + "_" + str(model.width) + "_" + str(model.depth) + "_" + str(test_num)
    writer(test_num,sampling_method,tot_mean,tot_out,generalization_error,training_error,model.width,model.depth,state)
    store_data([num_out,mean_nv,stdv_nv,list_dist_nv,mean_dist,output_test], ["num_out","mean_nv","stdv_nv","list_dist_nv","mean_dist","output_test"], folder_name)

    return train_nv, test_nv, generalization_error, training_error, output_test