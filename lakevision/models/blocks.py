"""
Reusable model blocks for Greenland supraglacial lake drainage classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FrontCNN(nn.Module):
    """
    Convolutional Neural Network (CNN) block for feature extraction from input image timestacks.
    Located at the front of the model.
    Each layer consists of Conv2d -> LeakyReLU -> MaxPool2d

    Args:
        in_channels (int): number of input channels (e.g., 4 for RGB and mask)
        base_channels (int): number of base channels for the first convolutional layer
        num_layers (int): number of convolutional layers (each with Conv2d -> LeakyReLU -> MaxPool2d)
        out_hw (tuple): output height and width after convolutions and pooling
            NOTE: pooling will happen if the number of layers doesn't reduce the spatial dimensions to out_hw
        pool (str): type of pooling to use (select from 'max', 'avg', or 'none')

    Input:
        x: [B, T, C, H, W] tensor of input image timestacks
            B: batch size
            T: time steps
            C: channels (e.g., 4 for RGB and mask)
            H: height
            W: width
    
    Output:
        [B, T, C_out, H_out, W_out] tensor of extracted features after convolutional layers
            C_out = base_channels * (2 ** (num_layers - 1))
            H_out = out_hw[0]
            W_out = out_hw[1]

    Example 1:
        NOTE: (results in a 32x32 output spatial dimension; CLSTM would be used after)
        >>> frontcnn = FrontCNN(in_channels=4, base_channels=8, num_layers=3, out_hw=(32,32), pool='max')
        >>> x = torch.randn(16, 153, 4, 512, 512) # example input tensor [B=16, T=153, C=4, H=512, W=512]
        >>> out = frontcnn(x) # output tensor [B=16, T=153, C_out=32, H_out=32, W_out=32]
    
    Example 2:
        NOTE: (results in a 1x1 output spatial dimension; regular LSTM would be used after)
        >>> frontcnn = FrontCNN(in_channels=4, base_channels=8, num_layers=4, out_hw=(1,1), pool='avg')
        >>> x = torch.randn(8, 153, 512, 512) # example input tensor [B=8, T=153, C=4, H=512, W=512]
        >>> out = frontcnn(x) # output tensor [B=8, T=152, C_out=64, H_out=1, W_out=1]
    
    """
    def __init__(self, in_channels=4, base_channels=8, num_layers=3, out_hw=(32,32), pool='max'):
        super(FrontCNN, self).__init__()
        
        if pool not in ['max', 'avg', 'none']:
            raise ValueError(f"pool must be one of: 'max', 'avg', or 'none', but got '{pool}'")

        layers = []
        C_in = in_channels
        C_out = base_channels

        for i in range(num_layers):
            layers.append(nn.Conv2d(C_in, C_out, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2)) # downsample by 2
            C_in = C_out
            C_out *= 2 # double channels each layer

        self.conv_block = nn.Sequential(*layers)
        self.output_channels = C_in # the last output channel after all layers

        # store out_hw and pool type for conditional pooling in forward pass
        self.out_hw = out_hw
        self.pool = pool

    def forward(self, x):
        """
        Forward pass of the FrontCNN block.
        Applies convolutional layers and pools conditionally to achieve desired output spatial dimensions.
        """
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W) # merge batch and time dimensions -> e.g., [B*T, 4, 512, 512]
        x = self.conv_block(x) # apply convolutional layers -> e.g., [B*T, C=base_channels*(2**num_layers-1), H_out, W_out]
        _, C_out, H_conv, W_conv = x.shape
        
        # conditional pooling:
        target_h, target_w = self.out_hw
        if (H_conv, W_conv) != (target_h, target_w):
            # need to pool
            if self.pool == 'max':
                x = F.adaptive_max_pool2d(x, self.out_hw)
            elif self.pool == 'avg':
                x = F.adaptive_avg_pool2d(x, self.out_hw)
            else:
                # If pool='none' but dimensions don't match, raise an error
                raise ValueError(
                    f"Output dimensions {(H_out, W_out)} do not match target {self.out_hw} "
                    f"and pool='none' was specified. Either change num_layers or set pool to 'max' or 'avg'."
                )
        
        # reshape back to [B, T, C_out, H_out, W_out]
        _, C_out, H_out, W_out = x.shape
        x = x.view(B, T, C_out, H_out, W_out)

        return x
    