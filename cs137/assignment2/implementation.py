"""
This problem is modified from a problem in Stanford CS 231n assignment 1. 
In this problem, we implement the neural network with pytorch instead of numpy
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader


# This class is implemented for you to hold datasets. It can be used to initialize a data loader or not. You may choose to use it or not.  

class MyDataset(Dataset):
    def __init__(self, x, y, pr_type):
        self.x = torch.tensor(x, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = (torch.long if pr_type == "classification"
                                                     else torch.float32))
    def __len__(self):
        return self.x.size()[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# NOTE: you need to complete two classes and one function below. Please see instructions marked by "TODO".   

class DenseLayer(nn.Module):
    """
     Implement a dense layer
    """
    def __init__(self, input_dim, output_dim, activation, reg_weight, init='autograder'):

        """
        Initialize weights of the DenseLayer. 
        args:
            input_dim: integer, the dimension of the input layer
            output_dim: integer, the dimension of the output layer
            activation: string, can be 'linear', 'relu', 'tanh', 'sigmoid', or 'softmax'. 
                        It specifies the activation function of the layer
            reg_weight: the regularization weight/strength, the lambda value in a regularization 
                        term 0.5 * \lambda * ||W||_2^2
            param_init: this input is used by autograder. Please do NOT touch it. 
        """

        super(DenseLayer, self).__init__()

        # set initial values for weights and bias terms. 
        param_init = dict(W=None, b=None)

        if init == 'autograder':
            np.random.seed(137)
            param_init['W'] = np.random.random_sample((input_dim, output_dim)) 
            param_init['b'] = np.random.random_sample((output_dim, )) 
        else:
            # TODO: please do your own initialization using the same dimension as above
            # Note: bad initializations may lead to bad performance later
            # np.random.seed(137)
            # param_init['W'] = 1/np.sqrt(output_dim) * np.random.randn(input_dim, output_dim)
            # param_init['b'] = np.random.randn(output_dim,)
            param_init['W'] = np.random.random_sample((input_dim, output_dim))
            param_init['b'] = np.random.random_sample((output_dim,))


        self.W = nn.Parameter(torch.tensor(param_init['W'], dtype = torch.float32))
        self.b = nn.Parameter(torch.tensor(param_init['b'], dtype = torch.float32))
        
        # TODO: record input arguments and initialize other necessary variables
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.reg_weight = reg_weight
        #self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, inputs):
        """
        This function implement the `forward` function of the dense layer
        """
        #outputs = self.linear(inputs)
        outputs = torch.matmul(inputs, self.W) + self.b
        if self.activation == 'linear':
            pass
        elif self.activation == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        elif self.activation == 'tanh':
            outputs = torch.tanh(outputs)
        elif self.activation == 'relu':
            outputs = torch.relu(outputs)
        elif self.activation == 'softmax':
            outputs = torch.nn.functional.softmax(outputs, dim=1)
        return outputs
        # TODO: implement the linear transformation
        #outputs = self.linear(inputs)
        # TODO: implement the activation function
        #outputs = self.activation(outputs)
        #return outputs

    
class Feedforward(nn.Module):

    """
    A feedforward neural network. 
    """

    def __init__(self, input_size, depth, hidden_sizes, output_size, reg_weight, task_type):

        """
        Initialize the model. This way of specifying the model architecture is clumsy, but let's use this straightforward
        programming interface so it is easier to see the structure of the program. Later when you program with torch 
        layers, you will see more precise approaches of specifying network architectures.  

        args:
          input_size: integer, the dimension of the input.
          depth:  integer, the depth of the neural network, or the number of connection layers. 
          hidden_sizes: list of integers. The length of the list should be one less than the depth of the neural network.
                        The first number is the number of output units of first connection layer, and so on so forth.
          output_size: integer, the number of classes. In our regression problem, please use 1. 
          reg_weight: float, The weight/strength for the regularization term.
          task_type: string, 'regression' or 'classification'. The task type. 
        """

        super(Feedforward, self).__init__()

        # Add a condition to make the program robust
        if not (depth - len(hidden_sizes)) == 1:
            raise Exception("The depth (%d) of the network should be 1 larger than `hidden_sizes` (%d)." % (depth, len(hidden_sizes)))
        self.input_size = input_size
        self.hidden_size = hidden_sizes
        self.depth = depth
        self.reg_weight = reg_weight
        self.output_size = output_size
        # TODO: install all connection layers except the last one

        #self.layers = nn.ModuleList([DenseLayer(self.input_size,self.hidden_size[0],'relu',self.reg_weight)])
        # if self.depth > 2:
        #     self.layers = nn.ModuleList([DenseLayer(self.input_size, self.hidden_size[0], 'relu', self.reg_weight)])
        #     for i in range(1,self.depth-1):
        #         self.layers.append(DenseLayer(self.hidden_size[i-1],self.hidden_size[i],'relu',self.reg_weight))
        # elif self.depth == 2:
        #     self.layers = nn.ModuleList([DenseLayer(self.input_size, self.hidden_size[0], 'relu', self.reg_weight)])
        #
        # # TODO: decide the last layer according to the task type
        # if task_type == 'regression':
        #     # don't need to use any activation function
        #     l2 = DenseLayer(hidden_sizes[self.depth-2],self.output_size,'relu',self.reg_weight)
        #     self.layers.append(l2)
        # elif task_type == 'classification':
        #     # use softmax function to perform binary classification task
        #     l2 = DenseLayer(hidden_sizes[self.depth-2],self.output_size,'softmax',self.reg_weight)
        #     self.layers.append(l2)
        self.layers = nn.ModuleList(
            [DenseLayer(self.input_size, self.hidden_size[0], activation='tanh', reg_weight=self.reg_weight)])
        for i in range(0, self.depth - 2):
            # self.layers.append(self.dropout_lay)
            self.layers.append(DenseLayer(self.hidden_size[i], self.hidden_size[i + 1], activation='tanh',
                                          reg_weight=self.reg_weight))

        if (task_type == 'regression'):
            last_layer = DenseLayer(self.hidden_size[self.depth - 2], self.output_size, activation='linear',
                                    reg_weight=self.reg_weight)
            self.layers.append(last_layer)
        if (task_type == 'classification'):
            last_layer = DenseLayer(self.hidden_size[self.depth - 2], self.output_size, activation='softmax',
                                    reg_weight=self.reg_weight)
            self.layers.append(last_layer)


    def forward(self, inputs):
        """
        Implement the forward function of the network. 
        """
        #TODO: apply the network function to the input; and also sum all regularization terms to get a single regularization term
        # out = inputs
        # for i, l in enumerate(self.layers):
        #     out = l(out)
        # inputs = inputs
        # for i in range(len(self.layers)):
        #     out = self.layers[i](inputs)
        #     inputs = out
        out = inputs
        for f in self.layers:
            out = f(out)
        return out
    
    def calculate_reg_term(self):
        """
        Compute a regularization term from all model parameters  

        args:
        """

        # TODO: computer the regularization term from all connection weights 
        # Note: there is a convenient alternatives for L2 norm: using weight decay. here we consider a general approach, which
        # can calculate different types of regularization methods. 
        reg_term = 0.0
        # TODO: computer the regularization term from all connection weights
        for f in self.layers:
            reg_term = reg_term + 1/2 * torch.sum(torch.square(f.W))*self.reg_weight
        return reg_term
        # for layer in self.layers:
        #     reg_term += torch.sum(torch.square(layer.W))
        # reg_term = 0.5 * reg_term
        # return reg_term


def train(x_train, y_train, x_val, y_val, depth, hidden_sizes, reg_weight, num_train_epochs, task_type, batch_size=16,
          lr=0.02):
    """
    Train this neural network using stochastic gradient descent.

    args:
      x_train: `np.array((N, D))`, training data of N instances and D features.
      y_train: `np.array((N, C))`, training labels of N instances and C fitting targets
      x_val: `np.array((N1, D))`, validation data of N1 instances and D features.
      y_val: `np.array((N1, C))`, validation labels of N1 instances and C fitting targets
      depth: integer, the depth of the neural network
      hidden_sizes: list of integers. The length of the list should be one less than the depth of the neural network.
                    The first number is the number of output units of first connection layer, and so on so forth.

      reg_weight: float, the regularization strength.
      num_train_epochs: the number of training epochs.
      task_type: string, 'regression' or 'classification', the type of the learning task.
    """

    # TODO: set up dataloaders for the training set and the test set. Please check the DataLoader class. You will
    #  need to specify your batch sizes there.
    train_dataset = MyDataset(x_train, y_train, task_type)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = MyDataset(x_val, y_val, task_type)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # TODO: initialize a model with the Feedforward class
    input_size = x_train.shape[1]  # feature number D
    # out
    if task_type == 'regression':
        output_size = 1
    if task_type == 'classification':
        output_size = 10
    model = Feedforward(input_size, depth, hidden_sizes, output_size, reg_weight, task_type)

    # TODO: initialize an opimizer. You can check the documentation of torch.optim.SGD, but you can certainly try more advanced
    # optimizers
    # optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(model.parameters())
    if (task_type == 'regression'):
        loss_func = nn.MSELoss()
    if (task_type == 'classification'):
        loss_func = nn.CrossEntropyLoss()

    history = {"loss": [],  "val_loss": [],"accuracy": []}
    for epoch in range(num_train_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i,data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            # loss_each.requires_grad = True
            reg = reg_weight * model.calculate_reg_term()
            loss_total = loss + reg
            loss_total.backward()
            optimizer.step()
            running_loss = running_loss + loss_total.item()
            #print('loss becomes ', running_loss, ' after ', epoch, ' iterations.')

        #train_size = train_loader.__len__()
        avg_loss = running_loss / i
        history['loss'].append(avg_loss)
        #print(avg_loss)

        total_val_loss = 0
        val_size = 0
        correct = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, targets = data
                outputs = model(inputs)
                loss = loss_func(outputs, targets)
                total_val_loss = total_val_loss + loss.item()
                # accuracy = (outputs.argmax(1) == targets).sum()
                val_size += inputs.shape[0]
                y_pred = outputs.argmax(1)
                correct += (y_pred == targets).sum()
            acc = correct/val_size
            l = total_val_loss/val_size
            print('the epoch is ',epoch)
            print('The test accuracy is ', correct/val_size)
            print('the val_loss is ',total_val_loss/val_size)
            history['val_loss'].append(l)
            history['accuracy'].append(acc)
    return model, history

depth = 4
hidden_sizes = [15,15,15]
reg_weight = 0.01
num_train_epochs = 250
batch_size = 100
#
#regression
# def target_func(x):
#     y = np.sin(1 / x)
#     return y
# # initialize training and validation sets.
# np.random.seed(42)
# x_train = np.power(np.random.random_sample([2000, 1]), 4) + 0.05
# y_train = target_func(x_train)
#
# x_val = np.power(np.random.random_sample([2000, 1]), 4) + 0.05
# y_val = target_func(x_val)
#
# model,history = train(x_train, y_train, x_val, y_val,
#                       depth=depth,
#                       hidden_sizes=hidden_sizes,
#                       reg_weight=reg_weight,
#                       num_train_epochs=num_train_epochs,
#                       task_type='regression',
#                       batch_size=batch_size)
#

#classification
# from torchvision import datasets as dts
# from torchvision.transforms import ToTensor
#
# def transform(x):
#     return ToTensor()(x).flatten()
#
#
# traindt = dts.MNIST(
#     root = 'data',
#     train = True,
#     transform = transform,
#     download = True,
# )
# testdt = dts.MNIST(
#     root = 'data',
#     train = False,
#     transform = transform
# )
#
# x_tr   = traindt.data.numpy().reshape(-1, 28 * 28)
# x_test = testdt.data.numpy().reshape(-1, 28 * 28)
# y_tr   = traindt.targets.numpy()
# y_test = testdt.targets.numpy()
#
# import torch.nn.functional as F
# from sklearn.model_selection import train_test_split
#
# # separate a validation set
# x_train, x_val, y_train, y_val = train_test_split(x_tr, y_tr,
#                                                   train_size=0.8,
#                                                   stratify=y_tr)
#
# print('Shape of training input: ', x_train.shape)
# print('Shape of training labels: ', y_train.shape)
# print('Shape of validation input: ', x_val.shape)
# print('Shape of validation labels: ', y_val.shape)
# print('Shape of test input: ', x_test.shape)
# print('Shape of test labels: ', y_test.shape)
# print('Number of channels: ', np.max(y_train) + 1)
#
# #from solutions import train
#
# model, history = train(x_train,
#                        y_train,
#                        x_val,
#                        y_val,
#                        depth=2,
#                        hidden_sizes=[30],
#                        reg_weight=0.01,
#                        num_train_epochs=200,
#                        task_type='classification',
#                       lr=0.0001)