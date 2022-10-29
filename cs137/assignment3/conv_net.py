################################################################################################
from torch.nn import Sequential
from torch.nn import Conv2d, MaxPool2d, Dropout, Flatten, Linear, ReLU, BatchNorm2d, Sequential, AvgPool1d, Softmax, LazyLinear

# NOTE: you should NOT import anything else from torch or other deep learning packages. It 
# means you need to construct a CNN using these layers. If you need other layers, you can ask us 
# first, but you CANNOT use existing models such as ResNet from tensorflow. 
################################################################################################

def ConvNet(**kwargs):
    """
    Construct a CNN using `torch`: https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html. 
    """
    
    # TODO: implement your own model
    # This is a very simple model as an example, which sums up all pixels and does classification

    # model = Sequential(*[Conv2d(3, 64, [3, 3]),  # Input channel is 3, output channel is 64
    #                      BatchNorm2d(64),
    #                      ReLU(),
    #                      MaxPool2d(2),
    #                      Flatten(2, -1),  # N,C,H, W => N,C, H * W
    #                      AvgPool1d(5),  # N, C, L => N, C, L'
    #                      Flatten(1, -1),  # N, C, L' => N, L''
    #                      LazyLinear(3),
    #                      ])
    #my model
    #alexnet
    model = Sequential(
        Conv2d(3, 96, kernel_size=11, stride=4, padding=1), ReLU(),
        MaxPool2d(kernel_size=3, stride=2),
        Conv2d(96, 256, kernel_size=5, padding=2), ReLU(),
        MaxPool2d(kernel_size=3, stride=2),
        Conv2d(256, 384, kernel_size=3, padding=1), ReLU(),
        Conv2d(384, 384, kernel_size=3, padding=1), ReLU(),
        Conv2d(384, 256, kernel_size=3, padding=1), ReLU(),
        MaxPool2d(kernel_size=3, stride=2),
        Flatten(),
        Linear(6400, 200), ReLU(),
        Dropout(p=0.5),
        Linear(200, 200), ReLU(),
        Dropout(p=0.5),
        Linear(200, 3))
    return model


import torch
model = ConvNet()
X = torch.randn(1, 3, 224, 224)
for layer in model:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)

































####################################################################################################
# This is the end of this file. Please do not alter or remove the variable below; otherwise you will 
# get zero point for the entire assignment. 
DSVGDES = "63744945093b4af559797cca6cbec618"
####################################################################################################
