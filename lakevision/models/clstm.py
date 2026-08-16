"""
Convolutional LSTM (CLSTM) module for spatial-temporal sequence processing.

CLSTM extends standard LSTM to handle spatial data by replacing matrix multiplications with convolutions.
This preserves spatial structure throughout the recurrent computation.

Reference:
    Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting", NIPS 2015
"""
import torch
import torch.nn as nn

class CellCLSTM(nn.Module):
    """
    Convolutional LSTM cell.

    Unlike standard LSTM cells, which use multiplication, CLSTM cell uses
    convolutional operations to preserve spatial information throught recurrent computation.
    Each gate operates on spatial feature maps rather than flattened vectors.
    
    Args:
        input_channels (int): number of channels feature channels
        hidden_channels (int): number of hidden state channels
        kernel_size (int): size of the convolutional kernel (default: 3, must be odd)

    Input:
        x: [B, C_in, H, W] current timestep input tensor
        h_prev: [B, C_hidden, H, W] previous hidden state
        c_prev: [B, C_hidden, H, W] previous cell state

    Output:
        h: [B, C_hidden, H, W] new hidden state
        c: [B, C_hidden, H, W] new cell state

    Gates:
        i (input gate): controls what new information to store
        f (forget gate): controls what information to discard
        o (output gate): controls what to output from cell state
        g (cell gate): candidate values to add to cell state

    Example:
        >>> cell = CellCLSTM(input_channels=3, hidden_channels=64, kernel_size=3)
        >>> x = torch.randn(16, 32, 64, 64) # input tensor [B=16, C_in=32, H=64, W=64]
        >>> h = torch.zeros(16, 64, 64, 64) # previous hidden state [B=16, C_hidden=64, H=64, W=64]
        >>> c = torch.zeros(16, 64, 64, 64) # previous cell state [B=16, C_hidden=64, H=64, W=64]
        >>> h_new, c_new = cell(x, h, c) # new hidden and cell states
        >>> print(h_new.shape, c_new.shape) # both should be [16, 64, 64, 64]
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3,
                 forget_bias=0.0):
        super(CellCLSTM, self).__init__()

        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, but got {kernel_size}")

        padding = kernel_size // 2
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        # single convolution generates all 4 gates at once
        self.conv = nn.Conv2d(
            in_channels=input_channels+hidden_channels,
            out_channels=4*hidden_channels, # i, f, o, g gates
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # Forget-gate bias. At the default 0.0 the gate opens at sigmoid(0)=0.5,
        # so the cell state decays by ~half per step early in training — over
        # T=153 steps that erases any memory of a mid-season drainage event long
        # before the readout sees it. Initializing to 1.0 (sigmoid(1)=0.73) is the
        # standard fix (Jozefowicz et al. 2015). Kept at 0.0 by default so the
        # ESSD tags reproduce bit-for-bit.
        # Gate order matches the torch.chunk(gates, 4) split in forward: i, f, o, g.
        if forget_bias:
            with torch.no_grad():
                self.conv.bias[hidden_channels:2 * hidden_channels].fill_(forget_bias)

    def forward(self, x, h_prev, c_prev):
        """
        Forward pass through CLSTM cell.

        Args:
            x: [B, C_in, H, W] current timestep input tensor
            h_prev: [B, C_hidden, H, W] previous hidden state
            c_prev: [B, C_hidden, H, W] previous cell state

        Returns:
            h: [B, C_hidden, H, W] new hidden state
            c: [B, C_hidden, H, W] new cell state
        """
        # concatenate input and previous hidden state along channel dimension
        combined = torch.cat([x, h_prev], dim=1) # [B, C_in + C_hidden, H, W]

        # compute all gate activations with single convolution
        gates = self.conv(combined) # [B, 4*C_hidden, H, W]

        # split into four separate gates
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1) # each [B, C_hidden, H, W]

        # apply activation functions
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        o = torch.sigmoid(o)  # output gate
        g = torch.tanh(g)     # cell gate

        # update cell state
        c = f * c_prev + i * g # forget old + add new

        # compute new hidden state
        h = o * torch.tanh(c)

        return h, c

class CLSTM(nn.Module):
    """
    CLSTM module for processing spatial-temporal sequences.
    
    Process image sequences while preserving spatial structure through time
    by using convolutional LSTM cell (CellCLSTM) operations.  Unlike standard LSTMs,
    which flattens spatial dimensions, CLSTM maintains 2D structure at each timestep.

    Args:
        input_channels (int): number of input feature channels
        hidden_channels (int): number of hidden state channels
        kernel_size (int): convolutional kernel size (default: 3, must be odd)
        return_sequence (bool): if True, return hidden state for all timesteps;
                                if False, return only final hidden state
                                (default: True)
    
    Input:
        x: [B, T, C_in, H, W] tensor of image sequences
            B: batch size
            T: time steps
            C_in: input channels
            H: height
            W: width
    
    Output:
        if return_sequence=True: [B, T, C_hidden, H, W] tensor of hidden states for all timesteps
        if return_sequence=False: [B, C_hidden, H, W] tensor of final hidden state only

    Example 1:
        (return all timesteps)
        >>> clstm = CLSTM(input_channels=32, hidden_channels=64)
        >>> x = torch.randn(16, 153, 32, 64, 64) # input tensor [B=16, T=153, C_in=32, H=64, W=64]
        >>> out = clstm(x) # output tensor [B=16, T=153, C_hidden=64, H=64, W=64]

    Example 2:
        (return final timestep only)
        >>> clstm = CLSTM(input_channels=32, hidden_channels=64, return_sequence=False)
        >>> x = torch.randn(16, 153, 32, 64, 64) # input tensor [B=16, T=153, C_in=32, H=64, W=64]
        >>> out = clstm(x) # output tensor [B=16, C_hidden=64, H=64, W=64]
    """
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size=3,
        return_sequence=True,
        forget_bias=0.0,
    ):
        super(CLSTM, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.return_sequence = return_sequence

        self.cell = CellCLSTM(input_channels, hidden_channels, kernel_size,
                              forget_bias=forget_bias)

    def forward(self, x):
        """
        Forward pass to process sequence through CLSTM.

        Args:
            x: [B, T, C_in, H, W] tensor of image sequences

        Returns:
            [B, T, C_hidden, H, W] if return_sequence=True
            [B, C_hidden, H, W] if return_sequence=False
        """
        B, T, _, H, W = x.shape

        # initialize hidden and cell states
        h, c = self.init_hidden(B, H, W, x.device)

        # process sequence
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:,t], h, c) # process timestep
            outputs.append(h)

        if self.return_sequence:
            # return all timesteps
            h_seq = torch.stack(outputs, dim=1) # [B, T, C_hidden, H, W]
            return h_seq
        else:
            # return only final timestep
            return h # [B, C_hidden, H, W]

    def init_hidden(self, B, H, W, device):
        """
        Initialize hidden and cell states to zeros.
        
        Args:
            B: batch size
            H: height
            W: width
            device: device to place tensors on

        Returns:
            h: [B, C_hidden, H, W] zero-initialized hidden state
            c: [B, C_hidden, H, W] zero-initialized cell state
        """
        return(
            torch.zeros(B, self.hidden_channels, H, W, device=device),
            torch.zeros(B, self.hidden_channels, H, W, device=device),
        )