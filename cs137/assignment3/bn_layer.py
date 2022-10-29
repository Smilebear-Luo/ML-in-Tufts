import torch
import torch.nn as nn

class BNLayer(nn.Module):
    """
    Implement the batch normalization layer. 
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1,training=True):
        """
        Please consult the documentation of batch normalization 
        (https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
        for the meaning of the arguments. 
        
        Here you are asked to implementing a simpler version of the layer. Here are differences. 
        1. This implementation will be applied to a CNN's feature map with format N,C,H,W. so a channel of a 
           feature shares a set of parameters. 

        2. The initializers both arrays, so you can use them directly. (tensorflow has special initializer 
           classes)
        """

        super(BNLayer, self).__init__()

        # TODO: please complete the initialization of the class and take all these options
        shape = (1, num_features, 1, 1)
        self.running_mean = torch.zeros(shape)
        self.running_var  = torch.ones(shape)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.momentum = momentum
        self.eps = eps
        self.training = True


    def forward(self, inputs):
        """
        Implement the forward calculation during training and inference (testing) stages. 
        """
        # TODO: Please implement the caluclation of batch normalization in training and testing stages

        # NOTE 1. you can use the binary flag `self.training` to check whether the layer works in the training or testing stage:
        # NOTE 2. you need to update the moving average of mean and variance so you can use later in the inference stage 
        if self.training == False:
            X_hat = (inputs - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        else:
            if len(inputs.shape) == 2:
                mean = inputs.mean(dim=0)
                var = ((inputs - mean) ** 2).mean(dim=0)
            else:
                mean = inputs.mean(dim=(0, 2, 3), keepdim=True)
                var = ((inputs - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            X_hat = (inputs - mean) / torch.sqrt(var + self.eps)
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        outputs = self.gamma * X_hat + self.beta
        return outputs

# from torch.nn import BatchNorm2d
# import numpy as np
# # initialize a tensor X and pretend that the tensor is a feature map in a CNN.
# # In this case, we do normalization over different channels. As we have mentioned in the
# # class, all pixels in a channel share the same normalization parameters.
#
# np.random.seed(137)
# N, C, H, W = 200, 5, 50, 60
# X = torch.tensor(np.random.randn(N, C, H, W) * 0.2 + 5, dtype = torch.float32)
#
# # initialize parameters for batch normalization
#
# momentum = 0.1
# epsilon = 0.01
#
# # batch_normalization with torch.
#
# torch_bn = BatchNorm2d(5, eps = epsilon, momentum = momentum, dtype = torch.float32)
#
# # we use the same parameters as torch batch normalization.
# my_bn = BNLayer(5, eps=epsilon, momentum=momentum,training=True)
#
#
# # compare computation results over different training batches.
#
# res_tf = torch_bn(X[0:10])
# res_my = my_bn(X[0:10])
#
# def mean_diff(x, y):
#     with torch.no_grad():
#         if isinstance(x, torch.Tensor):
#             x = x.numpy()
#         if isinstance(y, torch.Tensor):
#             y = y.numpy()
#         err = np.mean(np.abs(x - y))
#     return err
#
# print("Calculation from the first batch: the difference between the two implementations is ",
#           mean_diff(res_tf, res_my))
#
# # run batch normalization on the second batch
#
# res_tf = torch_bn(X[10:20])
# res_my = my_bn(X[10:20])
#
# print("Calculation from the second batch: the difference between the two implementations is ",
#           mean_diff(res_tf, res_my))
#
# # run batch normalization on the third batch
#
# res_tf = torch_bn(X[20:30])
# res_my = my_bn(X[20:30])
#
# print("Calculation from the third batch: the difference between the two implementations is ",
#           mean_diff(res_tf, res_my))
#
# # compare computation results over a testing batch
# # my_bn.eval()
# # torch_bn.eval()
#
# res_tf = torch_bn(X)
# my_bn = BNLayer(5, eps=epsilon, momentum=momentum,training=False)
# res_my = my_bn(X)
#
# print("Calculation from testing: the difference between the two implementations is ",
#        mean_diff(res_tf, res_my))


































####################################################################################################
# This is the end of this file. Please do not alter or remove the variable below; otherwise you will 
# get zero point for the entire assignment. 
DSVGDES = "63744945093b4af559797cca6cbec618"
####################################################################################################
