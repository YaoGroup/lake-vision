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
    
class ClassHeadMLP(nn.Module):
    """
    Multi-layer perceptron (MLP) head block for classification.

    Takes concatenated feature vectors from multiple sources (e.g., CLSTM output, or scalar LSTM output)
    and outputs final class logits.  Includes optional dropout for regularization.

    Args:
        input_dim (int): Input feature dimension (e.g., concatenated features from image and scalar sequences)
        hidden_dims (int, list of int, or None): Hidden layer dimension(s).
            - int: single hidden layer dimension(s)
            - list of int: multiple hidden layers with specified dimension
            - None: no hidden layer (direct linear projection)
            Examples: 64, [64, 32], [128, 64, 32], None
            (default: 64)
        num_classes (int): Number of output classes (default: 4 for ND/ED/LD/CD)
        dropout (float): Dropout probability. Set to 0.0 to disble. (default: 0.0)
        activation (str): Activation function to use. Options: 'relu', 'leakyrelu', 'gelu'. (default: 'relu')

        Input:
            x: [B, input_dim] tensor of concatenated features after (C)LSTM processing
                B: batch size
                input_dim: total feature dimension (e.g., clstm_hidden + slstm_hidden)
            
        Output:
            [B, num_classes] tensor of unnormalized (raw) class logits
            Apply softmax externally for probabilities: torch.softmax(logits, dim=1)
            Apply argmax for predictions: torch.argmax(logits, dim=1)

        Example 1:
            (single hidden layer)
            >>> head = ClassMLP(input_dim=80, hidden_dims=64, num_classes=4)
            >>> features = torch.randn(16, 80) # [batch=16, features=80]
            >>> logits = head(features) # [batch=16, num_classes=4]
            >>> probs = torch.softmax(logits, dim=1) # class probabilities
            >>> preds = torch.argmax(logits, dim=1) # class predictions

        Example 2:
            (multiple hidden layers)
            >>> head = ClassMLP(input_dim=80, hidden_dims=[64, 32])
            >>> # architecture: 80 -> 64 -> 32 -> 4
            >>> logits = head(features) # [batch=16, num_classes=4]
            
        Example 3:
            (with dropout)
            >>> head = ClassMLP(input_dim=80, hidden_dims=64, num_classes=4, dropout=0.5)
            >>> logits = head(features) # dropout applied during training

        Example 4:
            (single layer, no hidden layer)
            >>> head = ClassMLP(input_dim=80, hidden_dims=None, num_classes=4)
            >>> logits = head(features) # direct linear projection
    """
    def __init__(self,
                 input_dim,
                 hidden_dims=64,
                 num_classes=4,
                 dropout=0.0,
                 activation='relu'
                 ):
        super(ClassHeadMLP, self).__init__()

        # normalize hidden_dims to a list
        if hidden_dims is None:
            hidden_dims = []
        elif isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        elif not isinstance(hidden_dims, list):
            raise ValueError(f"hidden_dims must be an int, list of ints, or None, but got {type(hidden_dims)}")

        # select activation function
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leakyrelu': nn.LeakyReLU(inplace=True),
            'gelu': nn.GELU()
        }

        if activation.lower() not in activations:
            raise ValueError(f"activation must be one of: {list(activations.keys())}, but got '{activation}'")

        act_fn = activations[activation.lower()]

        # build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # add output layer (no activation or dropout after final layer)
        layers.append(nn.Linear(prev_dim, num_classes))
        self.fc = nn.Sequential(*layers)

        # store config for inspection
        # self.input_dim = input_dim
        # self.hidden_dims = hidden_dims
        # self.num_classes = num_classes
        # self.dropout = dropout
        # self.activation = activation

    def forward(self, x):
        """
        Forward pass through classification head.

        Args:
            x: [B, input_dim] tensor of input features

        Returns:
            [B, num_classes] tensor of class logits (unnormalized scores)
        """
        return self.fc(x)
    
class ScalarLSTM(nn.Module):
    """
    LSTM block for processing scalar time series sequences.

    Processes 1D sequences of scalar values (e.g., water area, cloudy tile fraction)
    using standar LSTM architecture. Unlike CLSTM, which preserves spatial structure,
    this processes temporal sequences of scalar features only.

    Args:
        hidden_dim (int): hidden state dimension of LSTM (default: 16)
        num_layers (int): number of stacked LSTM layers (default: 1)
        dropout (float) dropout probability between the LSTM layers (default: 0.0)
            NOTE: only applied if num_layers > 1

    Input:
        x: [B, T, 1] tensor of scalar time series sequences
            B: batch size
            T: time steps
            1: single scalar feature per timestep (e.g., lake water area)

    Output:
        [B, T, hidden_dim] tensor of hidden states at all timesteps
            B: batch size
            T: time steps
            hidden_dim: LSTM hidden state dimension
    
    Example 1:
        (water area sequence)
        >>> area_lstm = ScalarSeqsLSTM(hidden_dim=16)
        >>> area_seq = torch.randn(16, 153, 1) # [B=16, T=153, 1]
        >>> out = area_lstm(area_seq) # [B=16, T=153, hidden_dim=16]
    """
    def __init__(self, hidden_dim=16, num_layers=1, dropout=0.0):
        super(ScalarLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        """
        Process scalar sequence through LSTM.

        Args:
            x: [B, T, 1] tensor of scalar sequence
        
        Returns:
            [B, T, hidden_dim] tensor of hidden states at all timesteps
        """
        # verify input shape
        if x.shape[-1] != 1:
            raise ValueError(f"Expected input with last dimension = 1 (single scalar feature), "
                             f"but got shape {x.shape}. Reminder, use separate ScalarSeqsLSTM "
                             f"instances for each scalar feature."
                             )
        
        # initialize hidden and cell states
        h0 = torch.zeros(
            self.lstm.num_layers,
            x.size(0),
            self.lstm.hidden_size,
            device=x.device
        )
        c0 = torch.zeros(
            self.lstm.num_layers,
            x.size(0),
            self.lstm.hidden_size,
            device=x.device
        )
        
        # forward pass through LSTM
        output, (h_n, c_n) = self.lstm(x, (h0, c0))

        return output # [B, T, hidden_dim]


class GlobalPooling(nn.Module):
    """
    Global pooling to reduce spatial dimensions to feature vectors.
    
    Reduces 2D spatial dimensions (H × W) to a single value per channel
    using either average pooling, max pooling, or both. Commonly used
    before classification layers to convert spatial feature maps to vectors.
    
    Args:
        pool_type (str): Type of pooling to apply. Options:
            - 'avg': Average pooling (mean across spatial dimensions)
            - 'max': Max pooling (maximum across spatial dimensions)
            - 'both': Concatenates avg and max (doubles output channels)
            (default: 'avg')
    
    Input:
        x: [B, C, H, W] or [B, T, C, H, W] tensor
            B: batch size
            T: time steps (optional)
            C: channels
            H: height
            W: width
    
    Output:
        [B, C] or [B, T, C] if pool_type='avg' or 'max'
        [B, 2*C] or [B, T, 2*C] if pool_type='both'
    
    Example 1: Average pooling (default)
        >>> pool = GlobalPooling(pool_type='avg')
        >>> x = torch.randn(2, 32, 64, 64)  # [B, C, H, W]
        >>> out = pool(x)  # [2, 32]
    
    Example 2: Max pooling
        >>> pool = GlobalPooling(pool_type='max')
        >>> x = torch.randn(2, 32, 64, 64)
        >>> out = pool(x)  # [2, 32]
    
    Example 3: Both (concatenated avg and max)
        >>> pool = GlobalPooling(pool_type='both')
        >>> x = torch.randn(2, 32, 64, 64)
        >>> out = pool(x)  # [2, 64] - channels doubled!
    
    Example 4: With time dimension
        >>> pool = GlobalPooling(pool_type='avg')
        >>> x = torch.randn(2, 10, 32, 64, 64)  # [B, T, C, H, W]
        >>> out = pool(x)  # [2, 10, 32]
    
    Note:
        When pool_type='both', output channels are doubled because
        average and max pooled features are concatenated.
    """
    def __init__(self, pool_type='avg'):
        super(GlobalPooling, self).__init__()
        
        valid_types = ['avg', 'max', 'both']
        if pool_type not in valid_types:
            raise ValueError(
                f"pool_type must be one of {valid_types}, got '{pool_type}'"
            )
        
        self.pool_type = pool_type
    
    def forward(self, x):
        """
        Apply global pooling to reduce spatial dimensions.
        
        Args:
            x: [B, C, H, W] or [B, T, C, H, W] tensor
        
        Returns:
            [B, C] or [B, T, C] tensor (or [B, 2*C] / [B, T, 2*C] if pool_type='both')
        """
        # Check if time dimension is present
        has_time = (x.ndim == 5)
        
        if has_time:
            # Merge batch and time for processing
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
        
        # Apply pooling
        if self.pool_type == 'avg':
            # Average pooling: mean of all spatial locations
            pooled = F.adaptive_avg_pool2d(x, (1, 1))  # [B*T, C, 1, 1]
        
        elif self.pool_type == 'max':
            # Max pooling: maximum of all spatial locations
            pooled = F.adaptive_max_pool2d(x, (1, 1))  # [B*T, C, 1, 1]
        
        elif self.pool_type == 'both':
            # Both: concatenate avg and max features
            avg_pool = F.adaptive_avg_pool2d(x, (1, 1))  # [B*T, C, 1, 1]
            max_pool = F.adaptive_max_pool2d(x, (1, 1))  # [B*T, C, 1, 1]
            pooled = torch.cat([avg_pool, max_pool], dim=1)  # [B*T, 2*C, 1, 1]
        
        # Remove spatial dimensions (1x1)
        pooled = pooled.squeeze(-1).squeeze(-1)  # [B*T, C] or [B*T, 2*C]
        
        # Restore time dimension if needed
        if has_time:
            C_out = pooled.shape[1]
            pooled = pooled.reshape(B, T, C_out)  # [B, T, C] or [B, T, 2*C]
        
        return pooled