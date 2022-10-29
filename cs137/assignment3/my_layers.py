import numpy as np
def conv_forward(input, filters, bias, stride, padding):
    """
    An implementation of the forward pass of the convolutional operation. 
    Please consult the documentation of `torch.nn.functional.conv2d` 
    [link](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html)
    for the calculation and arguments for this operation. 
    
    We are considering a simpler case: the `input` is always in the format "NCHW". 
    We only consider two padding cases: "SAME" and "VALID". 

    """
    # TODO: Please implement the forward pass of the convolutional operation   
    # NOTE: you will need to compute a few sizes -- a handdrawing is useful for you to do the calculation.
    N_in, C_in, H_in, W_in = input.shape
    K_in, K_out, K_H, K_W = filters.shape
    stride_h, stride_w = stride
    if padding == "same":
        p_h = int((K_H - 1) / 2)
        p_h_pad = (K_H - 1) % 2
        p_w = int((K_W - 1) /2)
        p_w_pad = (K_W - 1) % 2
        H_out = H_in - K_H + p_h * 2 + p_h_pad + 1
        W_out = W_in - K_W + p_w * 2 + p_w_pad + 1
        x_pad = np.pad(input, ((0,0),(0,0),(p_h,p_h+p_h_pad),(p_w,p_w+p_w_pad)), 'constant')
        out = np.zeros((N_in, K_in, H_out, W_out))
    elif padding == "valid":
        H_out = int((H_in - K_H + stride_h)/stride_h)
        W_out = int((W_in -K_W + stride_w)/stride_w)
        x_pad = np.pad(input, ((0,0),(0,0),(0,0),(0,0)), 'constant')
        out = np.zeros((N_in, K_in, H_out, W_out))
    for n in range(N_in):
        for j in range(K_in):
            for h in range(H_out):
                for w in range(W_out):
                    height = h + K_H
                    width = w + K_W
                    x_slice = x_pad[n, :, h:height, w:width]
                    out[n, j, h, w] = np.sum(x_slice * filters[j,:,:,:]) + bias[j]
    return out

def max_pool_forward(input, ksize, stride, padding = "VALID"): # No need for padding argument here
    """
    An implementation of the forward pass of the max-pooling operation. 
    Please consult the documentation of `torch.nn.MaxPool2d` 
    [link](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
    for the calculation and arguments for this operation. 
    
    We are considering a simpler case: the `input` is always in the format "NCHW". 
    We only consider two padding cases: "SAME" and "VALID". 


    """

    # TODO: Please implement the forward pass of the max-pooling operation
    N_in,C_in,H_in,W_in = input.shape
    h_pool,w_pool = ksize
    stride_h, stride_w = stride
    H_out = 1 + int((H_in - h_pool)/stride_h)
    W_out = 1 + int((W_in - w_pool)/stride_w)
    out = np.zeros((N_in, C_in, H_out, W_out))
    for n in range(N_in):
        for c in range(C_in):
            for h in range(H_out):
                for w in range(W_out):
                        h_start= h * stride_h
                        h_end = h_start + h_pool
                        w_start = w * stride_w
                        w_end = w_start + w_pool
                        out[n, c, h, w] = np.max(input[n, c, h_start:h_end, w_start:w_end])
    return out

# x = np.random.randn(*[1, 5, 10, 20]).astype(np.float32)
#
#
# # The first test case:
# h_stride = 1
# w_stride = 1
# padding = "same"
# t_mod = nn.Conv2d(in_channels=5, out_channels = 4, kernel_size = (3,3), stride=[h_stride, w_stride], padding=padding)
# t_out = t_mod(torch.tensor(x, dtype = torch.float32))
# with torch.no_grad():
#     w = t_mod.weight.numpy()
#     b = t_mod.bias.numpy()
# out = conv_forward(input=x, filters=w, bias = b, stride=[h_stride, w_stride], padding=padding)
#
# with torch.no_grad():
#     print('Difference:', mean_diff(out, t_out))
#
# # The second test case:
# h_stride = 1
# w_stride = 1
# padding = "same"
# t_mod = nn.Conv2d(in_channels=5, out_channels = 4, kernel_size = (7,7), stride=[h_stride, w_stride], padding=padding)
# t_out = t_mod(torch.tensor(x, dtype = torch.float32))
# with torch.no_grad():
#     w = t_mod.weight.numpy()
#     b = t_mod.bias.numpy()
# out = conv_forward(input=x, filters=w, bias = b, stride=[h_stride, w_stride], padding=padding)
#
# with torch.no_grad():
#     print('Difference:', mean_diff(out, t_out))
#
#
# # The third test case:
# h_stride = 1
# w_stride = 1
# padding = "valid"
# t_mod = nn.Conv2d(in_channels=5, out_channels = 4, kernel_size = (7,3), stride=[h_stride, w_stride], padding=padding)
# t_out = t_mod(torch.tensor(x, dtype = torch.float32))
# with torch.no_grad():
#     w = t_mod.weight.numpy()
#     b = t_mod.bias.numpy()
# out = conv_forward(input=x, filters=w, bias = b, stride=[h_stride, w_stride], padding=padding)
#
# with torch.no_grad():
#     print('Difference:', mean_diff(out, t_out))
#
#
# # The third test case:
# w = np.random.randn(3, 3, 5, 4).astype(np.float32)
# h_stride = 1
# w_stride = 1
# padding = "valid"
#
# t_mod = nn.Conv2d(in_channels=5, out_channels = 4, kernel_size = (3, 3), stride=[h_stride, w_stride], padding=padding)
# t_out = t_mod(torch.tensor(x, dtype = torch.float32))
# with torch.no_grad():
#     w = t_mod.weight.numpy()
#     b = t_mod.bias.numpy()
# out = conv_forward(input=x, filters=w, bias = b, stride=[h_stride, w_stride], padding=padding)
#
# with torch.no_grad():
#     print('Difference:', mean_diff(out, t_out))

# np.random.seed(137)
# # shape is NCHW
# x = np.random.randn(2, 4, 10, 20).astype(np.float32)
#
# pool_height = 3
# pool_width = 3
# stride_h = pool_height
# stride_w = pool_width
#
# my_out = max_pool_forward(x, [pool_height, pool_width], stride=[stride_h, stride_w], padding='VALID')
# t_mod = nn.MaxPool2d([pool_height, pool_width], stride=[stride_h, stride_w])
# t_out = t_mod(torch.tensor(x, dtype = torch.float32))
#
# with torch.no_grad():
#     t_out = t_mod(torch.tensor(x, dtype = torch.float32))
#     print('Difference: ', mean_diff(my_out, t_out))
#
#
#
# pool_height = 4
# pool_width = 4
# stride_h = 2
# stride_w = 2
# my_out = max_pool_forward(x, [pool_height, pool_width], stride=[stride_h, stride_w], padding='VALID')
# t_mod = nn.MaxPool2d([pool_height, pool_width], stride=[stride_h, stride_w])
# t_out = t_mod(torch.tensor(x, dtype = torch.float32))
#
# with torch.no_grad():
#     t_out = t_mod(torch.tensor(x, dtype = torch.float32))
#     print('Difference: ', mean_diff(my_out, t_out))
#
#
# pool_height = 5
# pool_width = 4
# stride_h = 2
# stride_w = 1
#
# my_out = max_pool_forward(x, [pool_height, pool_width], stride=[stride_h, stride_w], padding='VALID')
# t_mod = nn.MaxPool2d([pool_height, pool_width], stride=[stride_h, stride_w])
# with torch.no_grad():
#     t_out = t_mod(torch.tensor(x, dtype = torch.float32))
#     print('Difference: ', mean_diff(my_out, t_out))
































####################################################################################################
# This is the end of this file. Please do not alter or remove the variable below; otherwise you will 
# get zero point for the entire assignment. 
DSVGDES = "63744945093b4af559797cca6cbec618"
####################################################################################################
